import torch
from switchtransformer import SwitchTransformer
from config import SwitchTransformerConfig
from dataloader import Dataloader
from trainer import train

config = SwitchTransformerConfig(
            d_model = 128,
            d_kv = 64,
            d_ff = 128,
            n_heads = 8,
            n_experts = 8,      
            n_layers = 4,  
            router_dtype=torch.float32,
            expert_capacity = 1,
            jitter_noise = 1e-2,
            ignore_padded_tokens = True,
            is_decoder = True,
            vocab_size = 50257,
            dropout_rate = 0.4,
            seq_len = 256,
            batch_size = 16,
            device  = 'cpu',
            file_path = 'shakespere.txt',
            optim = 'adam',
            lr = 1e-4,
            betas = (0.9, 0.95),
            eps = 1e-6,
            epochs = 1
            )

model = SwitchTransformer(config)
dataloader = Dataloader(config)
train(config, model)





