from utils.config_parser import get_args
from utils.tools import (
    del_files,
    EarlyStopping,
    adjust_learning_rate,
    vali,
    load_content,
    generate_pathname,
)
from utils.metrics import CG0, CGD, CGI, CG, CG_AVG
import torch
from accelerate import Accelerator, DeepSpeedPlugin
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm

from models import TimeLLM, TradingLLM

from data_provider.data_factory import data_provider

import time
import random
import numpy as np
import os
import json
import csv

os.environ["CURL_CA_BUNDLE"] = ""
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"


fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

args = get_args()

deepspeed_plugin = DeepSpeedPlugin(hf_ds_config="./ds_config_zero2.json")
accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)


# file = open("logs/example.txt", 'w')
for ii in range(args.itr):

    train_data, train_loader = data_provider(args, "train")
    vali_data, vali_loader = data_provider(args, "val")
    test_data, test_loader = data_provider(args, "test")
    # print(train_data.values)

    if args.model == "TradingLLM":
        model = TradingLLM.Model(args).float()
    else:
        model = TimeLLM.Model(args).float()

    # creating a unique name for the model
    setting = generate_pathname(args, ii)

    path = os.path.join(
        args.checkpoints, setting + "-" + args.model_comment
    )  # unique checkpoint saving path

    # save model arguments
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path + '/' + 'args', 'w+') as f:
        json.dump(args.__dict__, f, indent=2)

    res_header = ["Epoch", "Cost", "TrainLoss",
                  "ValiLoss", "TestLoss", "MAELoss", "TestCG", "TestCGD", "TestCGI", "ValiCG", "ValiCGD", "ValiCGI", "TrainCG"]

    csvres = open(path+'/results.csv', 'w+')
    reswriter = csv.writer(csvres)
    reswriter.writerow(res_header)

    # load prompt to args.content
    args.content = load_content(args)
    if not os.path.exists(path) and accelerator.is_local_main_process:
        os.makedirs(path)

    time_now = time.time()

    train_steps = len(train_loader)

    # each epoch we check if the result is the best and if it isn't after n consecutive steps,
    # we stop
    early_stopping = EarlyStopping(
        accelerator=accelerator, patience=args.patience)

    trained_parameters = []
    for p in model.parameters():
        if p.requires_grad is True:
            trained_parameters.append(p)

    model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)
    model_optim.param_groups[0]['lr'] = args.learning_rate

    # create scheduler

    if args.lradj == "COS":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            model_optim, T_max=20, eta_min=1e-8
        )
    else:
        scheduler = lr_scheduler.OneCycleLR(
            optimizer=model_optim,
            steps_per_epoch=train_steps,
            pct_start=args.pct_start,
            epochs=args.train_epochs,
            max_lr=args.learning_rate,
        )

    criterion = nn.MSELoss()
    mae_metric = nn.L1Loss()

    train_data, train_loader, vali_loader, test_loader, model, model_optim, scheduler = (
        accelerator.prepare(
            train_data, train_loader, vali_loader, test_loader, model, model_optim, scheduler
        )
    )

    for epoch in range(args.train_epochs):
        iter_count = 0
        train_loss = []

        model.train()
        epoch_time = time.time()
        train_cg_loss = []

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(
            enumerate(train_loader)
        ):
            
            iter_count += 1
            model_optim.zero_grad()
            # print("batch dimensions: ", batch_x.size())

            # accelerator.print(batch_x, batch_x_mark)

            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float().to(accelerator.device)
            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)

            # decoder input
            # acceleerate is using this somehow
            dec_inp = (
                torch.zeros_like(batch_y[:, -args.pred_len:, :])
                .float()
                .to(accelerator.device)
            )
            dec_inp = (
                torch.cat([batch_y[:, : args.label_len, :], dec_inp], dim=1)
                .float()
                .to(accelerator.device)
            )

            # encoder - decoder
            if args.output_attention:
                outputs = model(batch_x, batch_x_mark,
                                dec_inp, batch_y_mark)[0]
            else:
                outputs = model(batch_x, batch_x_mark,
                                dec_inp, batch_y_mark)
            
            

            f_dim = -1 if args.features == "MS" else 0
            last_vals = batch_x[:, -1, f_dim:]
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:]
            loss = criterion(outputs, batch_y)
            train_loss.append(loss.item())

            last_vals.detach()
            outputs.detach()
            batch_y.detach()
            train_cg_loss.append(CG(args.cg_value, last_vals, outputs, batch_y))
            

            
            if (i + 1) % 100 == 0:
                speed = (time.time() - time_now) / iter_count
                left_time = speed * \
                    ((args.train_epochs - epoch) * train_steps - i)
                accelerator.print(
                    "\tspeed: {:.4f}s/iter; left time: {:.4f}s, CG: {:.4f}".format(
                        speed, left_time, np.mean(train_cg_loss))
                )
                # print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()

            accelerator.backward(loss)
            model_optim.step()

            if args.lradj == "TST":
                adjust_learning_rate(
                    accelerator, model_optim, scheduler, epoch + 1, args, printout=False
                )
                scheduler.step()

        accelerator.print(
            "Epoch: {} cost time: {}".format(
                epoch + 1, time.time() - epoch_time)
        )
        train_loss = np.average(train_loss)
        vali_loss, vali_mae_loss, vali_metrics = vali(
            args, accelerator, model, vali_data, vali_loader, criterion, mae_metric
        )
        test_loss, test_mae_loss, test_metrics = vali(
            args, accelerator, model, test_data, test_loader, criterion, mae_metric
        )
        accelerator.print(
            "Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f} Test Loss: {3:.7f} MAE Loss: {4:.7f}  CG_train: {5:.7f} CG: {6:.7f}  CGI: {7:.7f} CG0_vali: {8:.7f} CGI_vali: {9:.7f} ".format(
                epoch + 1, train_loss, vali_loss, test_loss, test_mae_loss, np.mean(train_cg_loss), test_metrics.cg0, test_metrics.cgi, vali_metrics.cg0, vali_metrics.cgi
            )
        )
        reswriter.writerow([epoch+1, time.time() - epoch_time,
                           train_loss, vali_loss, test_loss, test_mae_loss, test_metrics.cg0, test_metrics.cgd, test_metrics.cgi, vali_metrics.cg0, vali_metrics.cgd, vali_metrics.cgi, np.mean(train_cg_loss)])
        csvres.flush()

        early_stopping(vali_loss, model, path)
        if early_stopping.early_stop:
            accelerator.print("Early stopping")
            break

        if args.lradj != "TST":
            if args.lradj == "COS":
                scheduler.step()
                accelerator.print(
                    "lr = {:.10f}".format(model_optim.param_groups[0]["lr"])
                )
            else:
                if epoch == 0:
                    args.learning_rate = model_optim.param_groups[0]["lr"]
                    accelerator.print(
                        "lr = {:.10f}".format(
                            model_optim.param_groups[0]["lr"])
                    )
                adjust_learning_rate(
                    accelerator, model_optim, scheduler, epoch + 1, args, printout=True
                )

        else:
            accelerator.print(
                "Updating learning rate to {}".format(
                    scheduler.get_last_lr()[0])
            )

accelerator.wait_for_everyone()

if accelerator.is_local_main_process:
    path = "./checkpoints"  # unique checkpoint saving path
    # del_files(path)  # delete checkpoint files
    accelerator.print("success delete checkpoints")
# file.close()
    csvres.close()
