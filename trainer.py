import torch
import numpy
import torch.nn as nn
from dataloader import Dataloader


max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073 # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)



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
            optim.lr = get_lr(i)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            running_loss.append(loss.item())
            optim.step()
        losses[f'epoch_{i}'] = running_loss
        print(f'epcohs:{i}/{config.epochs} | loss: {numpy.mean(running_loss):.4f} | ')



