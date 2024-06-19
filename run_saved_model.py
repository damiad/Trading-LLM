import torch
from accelerate import Accelerator, DeepSpeedPlugin
from torch import nn, optim
from torch.optim import lr_scheduler

from models import TradingLLM

from data_provider.data_factory import data_provider
import random
import numpy as np
import os
import json

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

from utils.tools import vali, load_content
from utils.config_parser import get_args
fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

args = get_args()
path = args.model_checkpoint_path

with open(path +'/args', 'r') as f:
    args.__dict__ = json.load(f)

args.model_checkpoint_path = path

deepspeed_plugin = DeepSpeedPlugin(hf_ds_config="./ds_config_zero2.json")
accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)

file = open("logs/example.txt", 'w')
for ii in range(1):
    train_data, train_loader = data_provider(args, 'train')
    test_data, test_loader = data_provider(args, 'test')

    if args.model == 'TradingLLM':
        model = TradingLLM.Model(args).float()
    else:
        raise ValueError('Model not supported')

    args.content = load_content(args)

    train_steps = len(train_loader)

    trained_parameters = []
    for p in model.parameters():
        if p.requires_grad is True:
            trained_parameters.append(p)

    model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)

    if args.lradj == 'COS':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=20, eta_min=1e-8)
    else:
        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=args.pct_start,
                                            epochs=args.train_epochs,
                                            max_lr=args.learning_rate)

    criterion = nn.MSELoss()
    mae_metric = nn.L1Loss()

    model.load_state_dict(torch.load(args.model_checkpoint_path + '/checkpoint', map_location=torch.device('cpu')))
    test_loader, model, model_optim, scheduler= accelerator.prepare(test_loader, model, model_optim, scheduler)    
    test_loss, test_mae_loss = vali(args, accelerator, model, test_data, test_loader, criterion, mae_metric)
    accelerator.print("Test Loss: {0:.7f} MAE Loss: {1:.7f}".format(test_loss, test_mae_loss))


accelerator.wait_for_everyone()
