import torch
import numpy
import torch.nn as nn
from dataloader import Dataloader
import math


max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 38146 # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens
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
    model.to(config.device)
    model.train()
    
    tr_dataloader = Dataloader(config, mode='train')
    val_dataloader = Dataloader(config, mode = 'val')

    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if param.ndim == 1 or "ln" in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)

    if config.optim == 'adam':
        optim = torch.optim.AdamW([
            {"params":decay, "weight_decay" :config.weight_decay,
             "params":no_decay, "weight_decay":0.0
            }
            ], lr = config.lr, betas = config.betas, eps = config.eps)
    else:
        optim = torch.optim.SGD(model.parameters(),lr = config.lr, weight_decay=config.weight_decay)

    log_metrics = {
        "tokens": [],
        "loss": [],
        'aux_loss':[],
        "lr": [],
        "grad_norm": [],
        "expert_usage":[],
        "router_entropy":[],
        "drop_rate":[],
        'load_imbalance':[]
    }

    tokens_seen = 0
    for i in range(1,config.epochs+1):
        for step in range(1, tr_dataloader.steps):
            optim.zero_grad()
            x,y = tr_dataloader.next_batch()
            out = model.forward(x,targets = y)
            loss = out['loss']
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            lr = get_lr(step)
            for pg in optim.param_groups:
                pg['lr'] = lr
            optim.step()
            
            tokens_seen += x.numel()
            log_metrics['loss'].append(loss.item())
            log_metrics['aux_loss'].append(out['cum_aux_loss'])
            log_metrics["grad_norm"].append(norm.item())
            log_metrics["lr"].append(lr)
            log_metrics["tokens"].append(tokens_seen)
            for layer_stats in out['router_stats']:
                log_expert_usage(layer_stats['selected_experts'],log_metrics)
                log_router_entropy(layer_stats['router_probs'],log_metrics)
                log_metrics['drop_rate'] = layer_stats['num_dropped']/(config.batch_size*config.seq_len)
            
            if step%config.log_interval == 0:
                print(f'epcohs:{i}/{config.epochs} | step: {step} | loss: {loss.item():.4f} | aux_loss: {out['cum_aux_loss']} | norm: {norm:.3f} | toekns: {tokens_seen/1e+6:.2f}M')
    return log_metrics



def log_expert_usage(selected_experts,log_metrics):
    batch_size, seq_len, n_experts = selected_experts.shape
    with torch.no_grad():
        selected_experts = selected_experts.view(-1, n_experts)
        load = selected_experts.sum(dim=0)
        load_frac = load / (batch_size*seq_len)
        log_metrics['expert_usage'].append(load_frac.detach())
        log_metrics['load_imbalance'].append(load_frac.var().detach())


def log_router_entropy(router_probs, log_metrics):
    batch_size, seq_len, n_experts = router_probs.shape
    with torch.no_grad():
        router_probs = router_probs.view(-1, n_experts)
        entropy = -(router_probs * torch.log(router_probs + 1e-9)).sum(dim=-1)
        mean_entropy = entropy.mean()
    log_metrics['router_entropy'].append(mean_entropy.detach())

        