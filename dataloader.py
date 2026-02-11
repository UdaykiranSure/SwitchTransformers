import torch
import tiktoken 
from config import SwitchTransformerConfig

class Dataloader():
    def __init__(self, config, tokens):
        self.batch_size = config.batch_size
        self.seq_len = config.seq_len
        self.device = config.device
        self.tokens = tokens
        self.current_pos = 0
        self.steps = len(self.tokens)//(self.batch_size*self.seq_len)


    def __iter__(self):
        return self
    
    def __next__(self):
        B,T = self.batch_size, self.seq_len
        buf = self.tokens[self.current_pos: self.current_pos+ B*T + 1]
        x = buf[:-1].view(B,T)
        y = buf[1:].view(B,T)
        self.current_pos += B*T
        if self.current_pos+ B*T > len(self.tokens):
            self.current_pos = 0
        return x,y


def load_and_Tokenize(file_path, device):
    with open(file_path, 'r') as f:
        text = f.read()
    enc = tiktoken.get_encoding('gpt2')
    tokens = enc.encode(text)
    tokens = torch.tensor(tokens, device = device)
    return tokens

def get_dataloader(config, split, val_ratio = 0.2):
    tokens = load_and_Tokenize(config.file_path, device = config.device)
    split_index = int(tokens.shape[0]*(1-val_ratio))
    if split == 'train':
        tokens = tokens[:split_index]
    else:
        tokens = tokens[split_index:]
    return Dataloader(config, tokens)



# config = SwitchTransformerConfig(1,1,1,16,8,'cpu')
# dataloader = Dataloader(config, 'shakespere.txt')
# config.batch_size[0]
# x,y = dataloader.next_batch()
# x.shape, y.shape
    