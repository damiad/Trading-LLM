import os
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 seasonal_patterns=None):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len,
                    12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 +
                    4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(
                lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(
                df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t',
                 seasonal_patterns=None):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 *
                    30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 *
                    30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(
                lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(
                df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 seasonal_patterns=None, to_remove=[], date_col='date', do_shift=False, seq_step=1):
        assert size != None
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.seq_step = seq_step

        assert flag in ['train', 'test', 'val', 'entire']
        type_map = {'train': 0, 'val': 1, 'test': 2, 'entire': 3}
        self.set_type = type_map[flag]
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.do_shift = do_shift

        self.to_remove = to_remove
        self.date_col = date_col

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - (self.seq_len +
                                           self.pred_len - 1)*self.seq_step 

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        df_raw.rename(columns={self.date_col: 'date'}, inplace=True)
        df_raw.drop(self.to_remove, axis=1, inplace=True)

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)

        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len * self.seq_step,
                    len(df_raw) - num_test - self.seq_len*self.seq_step, 0]
        border2s = [num_train, num_train + num_vali, len(df_raw), len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_data = df_raw[df_raw.columns[1:]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(
                lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(
                df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len
        s_end = s_begin + self.seq_len * self.seq_step
        r_begin = s_end - self.label_len * self.seq_step
        r_end = r_begin + (self.label_len + self.pred_len) * self.seq_step
        seq_x = self.data_x[s_begin:s_end:self.seq_step, feat_id:feat_id + 1]
        seq_y = self.data_y[r_begin:r_end:self.seq_step, feat_id:feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end:self.seq_step]
        seq_y_mark = self.data_stamp[r_begin:r_end:self.seq_step]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return self.tot_len * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_GBPCAD_hour(Dataset_Custom):
    def __init__(self, root_path, flag='train', size=None,
                 data_path='gbpcad_one_hour_202311210827.csv',
                 target='close', scale=True, timeenc=0, freq='h',
                 seasonal_patterns=None, to_remove=['id', 'provider', 'dayOfWeek', 'insertTimestamp', 'open', 'spread', 'usdPerPips', 'ask_volume', 'volume', 'ask_open', 'ask_low', 'ask_high', 'ask_close', 'ask_close', 'low', 'high'], 
                 date_col='barTimestamp', seq_step=1):

        super().__init__(root_path, flag=flag, size=size, data_path=data_path, target=target, scale=scale, timeenc=timeenc, freq=freq,
                         seasonal_patterns=seasonal_patterns, to_remove=to_remove, date_col=date_col, do_shift=True, seq_step=seq_step)


class Dataset_Sine_01(Dataset_Custom):
    def __init__(self, root_path, flag='train', size=None,
                 data_path='sine.csv',
                 target='target', scale=True, timeenc=0, freq='h',
                 seasonal_patterns=None, to_remove=[], date_col='date', seq_step=1):

        super().__init__(root_path, flag=flag, size=size, data_path=data_path, target=target, scale=scale, timeenc=timeenc, freq=freq,
                         seasonal_patterns=seasonal_patterns, to_remove=to_remove, date_col=date_col, seq_step=seq_step)

class Dataset_NUMSOLD_day(Dataset_Custom):
    def __init__(self, root_path, flag='train', size=None,
                 data_path='NUMSOLD-train.csv',
                 target='number_sold', scale=True, timeenc=0, freq='d',
                 seasonal_patterns=None, to_remove=['store','product'], date_col='Date', seq_step=1):

        super().__init__(root_path, flag=flag, size=size, data_path=data_path, target=target, scale=scale, timeenc=timeenc, freq=freq,
                         seasonal_patterns=seasonal_patterns, to_remove=to_remove, date_col=date_col, do_shift=True, seq_step=seq_step)

class Dataset_EURUSD_hour(Dataset_Custom):
    def __init__(self, root_path, flag='train', size=None,
                 data_path='EURUSD_ONE_HOUR_202311210826.csv',
                 target='close', scale=True, timeenc=0, freq='h',
                 seasonal_patterns=None, to_remove=['id', 'provider', 'dayOfWeek', 'insertTimestamp', 'open', 'spread', 'usdPerPips', 'ask_volume', 'volume', 'ask_open', 'ask_low', 'ask_high', 'ask_close', 'ask_close', 'low', 'high'], 
                 date_col='barTimestamp', seq_step=1):

        super().__init__(root_path, flag=flag, size=size, data_path=data_path, target=target, scale=scale, timeenc=timeenc, freq=freq,
                         seasonal_patterns=seasonal_patterns, to_remove=to_remove, date_col=date_col, do_shift=True, seq_step=seq_step)

class Dataset_AAPL(Dataset_Custom):
    def __init__(self, root_path, flag='train', size=None,
                 data_path='AAPL.csv',
                 target='Close', scale=True, timeenc=0, freq='d',
                 seasonal_patterns=None, to_remove=['Open', 'High', 'Low', 'Adj Close', 'Volume'], date_col='Date', seq_step=1):

        super().__init__(root_path, flag=flag, size=size, data_path=data_path, target=target, scale=scale, timeenc=timeenc, freq=freq,
                         seasonal_patterns=seasonal_patterns, to_remove=to_remove, date_col=date_col, seq_step=seq_step)
        
class Dataset_GBPTRY_hour(Dataset_Custom):
    def __init__(self, root_path, flag='train', size=None,
                 data_path='GBPTRY_ONE_HOUR.csv',
                 target='close', scale=True, timeenc=0, freq='h',
                 seasonal_patterns=None, to_remove=['open','low','high','ask_open','ask_close','ask_low','ask_high','usdPerPips'], 
                 date_col='barTimestamp', seq_step=1):

        super().__init__(root_path, flag=flag, size=size, data_path=data_path, target=target, scale=scale, timeenc=timeenc, freq=freq,
                         seasonal_patterns=seasonal_patterns, to_remove=to_remove, date_col=date_col, do_shift=True, seq_step=seq_step)
                 

class Dataset_ETHUSD_hour(Dataset_Custom):
    def __init__(self, root_path, flag='train', size=None,
                 data_path='ETHUSD_ONE_HOUR.csv',
                 target='close', scale=True, timeenc=0, freq='h',
                 seasonal_patterns=None, to_remove=["open","low","high","ask_open","ask_close","ask_low","ask_high","usdPerPips"], 
                 date_col='barTimestamp', seq_step=1):

        super().__init__(root_path, flag=flag, size=size, data_path=data_path, target=target, scale=scale, timeenc=timeenc, freq=freq,
                         seasonal_patterns=seasonal_patterns, to_remove=to_remove, date_col=date_col, do_shift=True, seq_step=seq_step)
        
class Dataset_BTCUSD_hour(Dataset_Custom):
    def __init__(self, root_path, flag='train', size=None,
                 data_path='BTCUSD_ONE_HOUR.csv',
                 target='close', scale=True, timeenc=0, freq='h',
                 seasonal_patterns=None, to_remove=['open','low','high','ask_open','ask_close','ask_low','ask_high','usdPerPips'], 
                 date_col='barTimestamp', seq_step=1):

        super().__init__(root_path, flag=flag, size=size, data_path=data_path, target=target, scale=scale, timeenc=timeenc, freq=freq,
                         seasonal_patterns=seasonal_patterns, to_remove=to_remove, date_col=date_col, do_shift=True, seq_step=seq_step)



class Dataset_US500_hour(Dataset_Custom):
    def __init__(self, root_path, flag='train', size=None,
                 data_path='ETHUSD_ONE_HOUR.csv',
                 target='close', scale=True, timeenc=0, freq='h',
                 seasonal_patterns=None, to_remove=["open","low","high","ask_open","ask_close","ask_low","ask_high","usdPerPips"], 
                 date_col='barTimestamp', seq_step=1):

        super().__init__(root_path, flag=flag, size=size, data_path=data_path, target=target, scale=scale, timeenc=timeenc, freq=freq,
                         seasonal_patterns=seasonal_patterns, to_remove=to_remove, date_col=date_col, do_shift=True, seq_step=seq_step)



class Dataset_WEATHER_hour(Dataset_Custom):
    def __init__(self, root_path, flag='train', size=None,
                 data_path='ETHUSD_ONE_HOUR.csv',
                 target="mean_temp", scale=True, timeenc=0, freq='h',
                 seasonal_patterns=None, to_remove=["cloud_cover","sunshine","global_radiation","max_temp","min_temp","precipitation","pressure","snow_depth"], 
                 date_col='barTimestamp', seq_step=1):

        super().__init__(root_path, flag=flag, size=size, data_path=data_path, target=target, scale=scale, timeenc=timeenc, freq=freq,
                         seasonal_patterns=seasonal_patterns, to_remove=to_remove, date_col=date_col, do_shift=True, seq_step=seq_step)
        

class Dataset_ELECTRICITY_hour(Dataset_Custom):
    def __init__(self, root_path, flag='train', size=None,
                 data_path='electricity.csv',
                 target='target', scale=True, timeenc=0, freq='h',
                seasonal_patterns=None, 
                 date_col='date', seq_step=1):

        super().__init__(root_path, flag=flag, size=size, data_path=data_path, target=target, scale=scale, timeenc=timeenc, freq=freq,
                         seasonal_patterns=seasonal_patterns, date_col=date_col, do_shift=True, seq_step=seq_step)