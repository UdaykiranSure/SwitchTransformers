import torch
import tiktoken 
from config import SwitchTransformerConfig

class Dataloader():
    def __init__(self, config):
        self.batch_size = config.batch_size
        self.seq_len = config.seq_len
        self.device = config.device
        self.tokens = self._load_and_Tokenize(config.file_path)
        self.current_pos = 0
        self.steps = len(self.tokens)//(self.batch_size*self.seq_len)

    def _load_and_Tokenize(self, file_path):
        with open(file_path, 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        tokens = torch.tensor(tokens, device = self.device)
        return tokens
    
    def next_batch(self):
        B,T = self.batch_size, self.seq_len
        buf = self.tokens[self.current_pos: self.current_pos+ B*T + 1]
        x = buf[:-1].view(B,T)
        y = buf[1:].view(B,T)
        self.current_pos += B*T
        if self.current_pos+ B*T > len(self.tokens):
            self.current_pos = 0
        return x,y



# config = SwitchTransformerConfig(1,1,1,16,8,'cpu')
# dataloader = Dataloader(config, 'shakespere.txt')
# config.batch_size[0]
# x,y = dataloader.next_batch()
# x.shape, y.shape
    