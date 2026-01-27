import torch
import numpy
import torch.nn as nn
from dataloader import Dataloader

def train(config, model):
    dataloader = Dataloader(config)

    if config.optim == 'adam':
        optim = torch.optim.AdamW(model.parameters(), lr = config.lr, betas = config.betas, eps = config.eps)
    else:
        optim = torch.optim.SGD(model.parameters(),lr = config.lr)

    losses = {}
    for i in range(1,config.epochs+1):
        running_loss = []
        for step in range(1, dataloader.steps):
            optim.zero_grad()
            x,y = dataloader.next_batch()
            out, loss = model.forward(x,targets = y)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            running_loss.append(loss.item())
            optim.step()
        losses[f'epoch_{i}'] = running_loss
        print(f'epcohs:{i}/{config.epochs} | loss: {numpy.mean(running_loss):.4f} | ')



