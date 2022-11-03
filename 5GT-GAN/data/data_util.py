import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from pytorch_lightning import LightningDataModule
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm
import os


class TrafficDataModule(LightningDataModule):
    def __init__(self, seq_len, data_paths, one_hot=False,
                 batch_size=64):
        super(TrafficDataModule, self).__init__()
        self.label = None
        self.data = None
        self.cols = None
        self.scaled_df = None
        self.seq_len = seq_len
        self.scalers = {}
        self.one_hot = one_hot
        self.batch_size = batch_size

        self.data_paths = data_paths

        self.conditions = []

        self.save_hyperparameters()

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.load_data(self.data_paths)
        else:
            self.load_data(self.data_paths, stage=stage)

    def load_data(self, data_paths, stage='fit'):
        df = pd.read_csv(os.getcwd() + '/' + data_paths)
        df = df[df.columns[~df.columns.str.contains('Unnamed')]]
        self.scaled_df = pd.DataFrame()
        self.cols = df.columns
        datas = []
        for i, col in enumerate(self.cols):
            scaler = MinMaxScaler(feature_range=(-1, 1))
            transformed = scaler.fit_transform(df[col].dropna().to_numpy()[:, np.newaxis])
            try:
                self.scaled_df[col] = transformed[:, 0]
            except ValueError as _:
                self.scaled_df = pd.concat([self.scaled_df, pd.DataFrame(columns=[col], data=transformed[:, 0])])
            data = sliding_window_view(transformed[:, 0], self.seq_len)
            datas.append(data)
            self.scalers[col] = scaler

        self.data = torch.tensor(np.concatenate(datas, axis=0), dtype=torch.float32)
        self.create_condition([len(data) for data in datas], stage=stage)

    def create_condition(self, len_datas, stage='fit'):
        labels = []
        print('Creating Conditions')
        for i, len_data in tqdm(enumerate(len_datas)):
            label = np.zeros(len(self.cols))
            label[i] = 1
            label = label.tolist()
            labels.append(torch.tensor([[label] * self.seq_len] * len_datas[i]))
            self.conditions.append(label)
        if stage is not 'inference':
            self.label = torch.cat(labels)

    def train_dataloader(self):
        return DataLoader(TrafficDataset(self.data, self.label), batch_size=self.batch_size, shuffle=True,
                          num_workers=2)


class TrafficDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item], self.label[item]
