from torch.utils.data import Dataset
import torch


class TSDataset(Dataset):
    def __init__(self, df, seq_size=500):
        self.data = df
        self.seq_size = seq_size
        self.step_size = seq_size // 20

    def __getitem__(self, idx):
        data = self.data.iloc[self.step_size * idx:  self.step_size * idx + self.seq_size].values[:, 2:]
        x = data[:, :5].astype('float32')
        y = data[:, 5:].astype('float32')
        return torch.tensor(x), torch.tensor(y)

    def __len__(self):
        return (len(self.data) - self.seq_size) // self.step_size
