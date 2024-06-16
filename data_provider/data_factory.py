from data_provider.data_loader import *

from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'gbpcad' : Dataset_GBPCAD_hour,
    'sine' : Dataset_Sine_01,
    'numsold' : Dataset_NUMSOLD_day,
    'eurusd' : Dataset_EURUSD_hour,
    'aapl' : Dataset_AAPL,
    'gbptry' : Dataset_GBPTRY_hour,
    'btcusd' : Dataset_BTCUSD_hour,
    'us500' : Dataset_US500_hour,
    'ethusd' : Dataset_ETHUSD_hour,
    'weather' : Dataset_WEATHER_hour,
    'electricity' : Dataset_ELECTRICITY_hour,

}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
    else:
        shuffle_flag = False


    drop_last = True
    batch_size = args.batch_size
    freq = args.freq

    if args.data == 'm4':
        drop_last = False

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        seasonal_patterns=args.seasonal_patterns,
        seq_step=args.seq_step
    )
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
