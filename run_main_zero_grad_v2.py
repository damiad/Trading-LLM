from utils.config_parser import get_args
from utils.tools import (
    del_files,
    EarlyStopping,
    adjust_learning_rate,
    vali,
    load_content,
    generate_pathname,
    format_arr
)
from utils.metrics import CGD, CGI, CG, CG_AVG, CG_arr
import torch
from accelerate import Accelerator, DeepSpeedPlugin
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm

from models import TradingLLM

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


for ii in range(args.itr):
    train_data, train_loader = data_provider(args, "entire")
    # entire_data, entire_loader = data_provider(args, "test")

    if args.model == "TradingLLM":
        model = TradingLLM.Model(args).float()
    else:
        raise ValueError("Unsupported model: {}".format(args.model))

    # creating a unique name for the model
    setting = generate_pathname(args, ii)
    path = os.path.join(
        args.checkpoints, setting + "-" + args.model_comment + "-zeroShot"
    )

    # save model arguments
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path + '/' + 'args', 'w+') as f:
        json.dump(args.__dict__, f, indent=2)

    # res_header = ["EntireLoss", "EntireMAELoss", "EntireCG"]
    res_header = ["EntireCG"]

    csvres = open(path+'/resultsZeroShot.csv', 'w+')
    reswriter = csv.writer(csvres)
    reswriter.writerow(res_header)

    # load prompt to args.content
    # args.content = load_content(args)
    if not os.path.exists(path) and accelerator.is_local_main_process:
        os.makedirs(path)

    train_steps = len(train_loader)

    trained_parameters = []
    for p in model.parameters():
        if p.requires_grad is True:
            trained_parameters.append(p)

    model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)
    
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

    # train_data, train_loader, entire_loader, model, model_optim, scheduler = (
    #     accelerator.prepare(
    #         train_data, train_loader, entire_loader, model, model_optim, scheduler
    #     )
    # )
    train_data, train_loader, model, model_optim, scheduler = (
        accelerator.prepare(
            train_data, train_loader, model, model_optim, scheduler
        )
    )

    train_loss = []
    model.train()
    epoch_time = time.time()
    train_cg_loss = []

    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(
            enumerate(train_loader)
        ):
            model_optim.zero_grad()
            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float().to(accelerator.device)
            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)

            # decoder input
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

            f_dim = 0
            last_vals = batch_x[:, -1, f_dim:]
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:]

            loss = criterion(outputs, batch_y)
            train_loss.append(loss.item())

            last_vals.detach()
            outputs.detach()
            batch_y.detach()
            train_cg_loss.append(CG_arr(last_vals, outputs, batch_y))

            # Without it our accuracy isn't such extreme!
            accelerator.backward(loss)
            model_optim.step()
            if args.lradj == "TST":
                adjust_learning_rate(
                    accelerator, model_optim, scheduler, epoch + 1, args, printout=False
                )
                scheduler.step()
    
    # entire_loss, entire_mae_loss, entire_metrics = vali(
    #         args, accelerator, model, entire_data, entire_loader, criterion, mae_metric
    #     )
    # reswriter.writerow([entire_loss, entire_mae_loss, entire_metrics.cg])
    reswriter.writerow([np.mean(train_cg_loss, axis=0)])
    
    csvres.flush()

accelerator.wait_for_everyone()

if accelerator.is_local_main_process:
    path = "./checkpoints"  # unique checkpoint saving path
    # del_files(path)  # delete checkpoint files
    accelerator.print("success delete checkpoints")
    csvres.close()
