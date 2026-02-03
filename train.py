import torch
from switchtransformer import SwitchTransformer
from config import SwitchTransformerConfig
from dataloader import Dataloader
from trainer import train
import json

config = SwitchTransformerConfig(
            d_model = 1024,
            d_kv = 256,
            d_ff = 512,
            n_heads = 16,
            n_experts = 8,      
            n_layers = 8,  
            router_dtype=torch.float32,
            expert_capacity = 1.25,
            jitter_noise = 1e-2,
            ignore_padded_tokens = True,
            is_decoder = True,
            vocab_size = 50257,
            dropout_rate = 0.4,
            seq_len = 2048,
            batch_size = 128,
            device  = 'cuda',
            file_path = 'shakespere.txt',
            optim = 'adam',
            lr = 1e-4,
            betas = (0.9, 0.95),
            eps = 1e-6,
            epochs = 1,
            weigh_decay=0.1,
            log_interval=50
            )

model = SwitchTransformer(config)
print(f'Total trainable parameters: {sum(p.numel() for p in model.parameters())}')
dataloader = Dataloader(config)
metrics = train(config, model)

with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent = 4)


# dataloader = Dataloader(config)
# for x,y in dataloader:
#     x.numel()
#     break
# model = SwitchTransformer(config)
# optim = torch.optim.AdamW(model.parameters(), 1e-3)
# optim.param_groups[0]['lr']
# for name,parameters in model.named_parameters():
#     print(name)