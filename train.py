from lib.dataset import TSDataset
import pandas as pd
import torch
from torch.utils.data import DataLoader
import collections
from lib.model import SimpleRNN
from catalyst.dl import SupervisedRunner
# from catalyst.dl.callbacks import  AUCCallback


if __name__ == '__main__':
    path = 'data.csv'
    data = pd.read_csv(path, nrows=1000000)
    train_size = int(0.7 * len(data))
    train = data[:train_size]
    test = data[train_size:]
    train_dataset = TSDataset(train)
    test_dataset = TSDataset(test)
    model = SimpleRNN(in_size=5, out_size=4, hidden_size=64)
    model = model.cuda()

    # experiment setup
    logdir = "./logdir"
    num_epochs = 20

    # data
    loaders = collections.OrderedDict()
    loaders["train"] = DataLoader(train_dataset, shuffle=False, batch_size=128, num_workers=4)
    loaders["valid"] = DataLoader(test_dataset, shuffle=False, batch_size=128, num_workers=4)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)

    callbacks = None  # [AUCCallback(), F1ScoreCallback()]

    # model runner
    runner = SupervisedRunner()

    # model training
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        logdir=logdir,
        num_epochs=num_epochs,
        callbacks=callbacks,
        verbose=True
    )
