#!/usr/bin/env python3

import numpy as np
import torch
from tqdm import tqdm
from utils import get_device, save_json
import torch.nn.functional as F
from argparse import ArgumentParser
from generator import *

DEVICE = get_device()

class LSTMModel(torch.nn.Module):
    def __init__(self, hidden_size=256):
        super().__init__()

        self.model_rnn = torch.nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=4,
            bidirectional=True,
            batch_first=True
        )
        self.model_dense = torch.nn.Sequential(
            torch.nn.Linear(2*hidden_size, 1)
        )
        self.loss = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=10e-6)

        self.to(DEVICE)

    def forward(self, x):
        seq_length = x.shape[0]
        x = x.reshape(1, seq_length, 1)

        # take (1) the output and (2) the first batch
        x = self.model_rnn(x)[0][0]
        # projection layer
        x = self.model_dense(x)
        x = x.reshape(seq_length)
        return x

    def check_hit(self, x1, x2):
        with torch.no_grad():
            x1 = torch.round(x1)
            x2 = torch.round(x2)
            return np.average([torch.equal(v1, v2) for v1, v2 in zip(x1, x2)])/len(x1)

    def train_loop(self, data_train, data_dev, prefix="", epochs=50):
        logdata = []
        for epoch in range(epochs):
            self.train(True)

            losses_train = []
            hits_train = []
            for arr_i, (arr_v, arr_sorted) in enumerate(tqdm(data_train)):
                arr_v = torch.Tensor(arr_v).to(DEVICE)
                arr_sorted = torch.Tensor(arr_sorted).to(DEVICE)

                out = self.forward(arr_v)
                loss = self.loss(out, arr_sorted)
                hits_train.append(self.check_hit(out, arr_sorted))

                # backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                losses_train.append(loss.detach().cpu().item())

                # one log step
                if arr_i % 1000 == 0:
                    losses_train = np.average(losses_train)
                    hits_train = np.average(hits_train)
                    # losses_dev = self.eval_dev(data_dev, encode_text)

                    # warning: train_loss is macroaverage, dev_loss is microaverage
                    logdata.append({
                        "epoch": epoch,
                        "train_loss": losses_train,
                        "train_hits": hits_train,
                        # "dev_loss": losses_dev,
                        # "dev_pp": 2**losses_dev
                    })

                    save_json(
                        f"computed/{prefix}.json",
                        logdata
                    )
                    losses_train = []
                    hits_train = []



if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("-p", "--prefix", default="")
    args.add_argument("-e", "--epochs", type=int, default=50)
    args.add_argument("--hidden-size", type=int, default=256)
    args = args.parse_args()

    data = [
        generate_0()
        for _ in range(int(10e3))
    ]
    data = [
        (v, sorted(v)) for v in data
    ]

    model = LSTMModel(hidden_size=args.hidden_size)
    model.train_loop(
        data[:-1000], data[-1000:],
        prefix=f"kenntheit",
        epochs=args.epochs
    )
