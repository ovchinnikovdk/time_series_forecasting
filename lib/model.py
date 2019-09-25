import torch


class SimpleRNN(torch.nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super(SimpleRNN, self).__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.num_layers = 4
        self.rnn = torch.nn.LSTM(input_size=self.in_size,
                                 hidden_size=self.hidden_size,
                                 num_layers=self.num_layers,
                                 batch_first=True,
                                 dropout=0.5)
        self.fc = torch.nn.Sequential(torch.nn.Linear(self.hidden_size, self.hidden_size),
                                      torch.nn.Dropout(0.4),
                                      torch.nn.Linear(self.hidden_size, self.out_size))

    def forward(self, x):
        x, h = self.rnn(x)
        x = self.fc(x)
        return x
