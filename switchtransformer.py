import torch
import torch.nn as nn
import torch.nn.functional as F

class Router(nn.Module):
    def __init__(self, config):
        """
        params:
        d_model: hidden vector size
        n_experts: number of routers
        expert_capacity: expert capacity
        jitter_noise: noise factor to add in the logits

        """
        super(Router, self ).__init__()
        self.d_model = config.d_model
        self.n_experts = config.n_experts
        self.expert_capacity = config.expert_capacity
        self.jitter_noise = config.jitter_noise
        self.ignore_padded_tokens = config.ignore_padded_tokens
        self.router_dtype = config.routet_dtype

        self.AuxLoss = AuxLoss(config)
        self.classifier = nn.Linear(self.d_model, self.n_experts)

    def forward(self, hidden_states: torch.Tensor):
        """
        inputs:
        hidden: hidden representations #(batch_size, seq_len, d_model)

        outputs:
        expert indices: one hot encoding of argmax probs of router for each token  #(batch_size, seq_len, n_experts)
        router logits: raw logits before softmaxing the probs #(batch_size, seq_len, n_experts)

        """

        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(self.router_dtype)

        # multiplicative jitter noise on the incoming representation
        if self.jitter_noise > 0:
            hidden_states *= torch.distributions.Uniform(1-self.jitter_noise, 1+self.jitter_noise).sample(hidden_states.shape)
        
        router_logits = self.classifier(hidden_states)  #(batch_size, seq_len, n_experts)
        router_probs = F.softmax(router_logits, dim=-1)

        top1_probs, expert_indices = torch.max(router_probs,dim=-1, keepdim=True)
        expert_indices = F.one_hot(expert_indices,sefl.n_experts).squeeze(-3)

        token_priority = torch.cumsum(expert_indices, dim=-2)
        expert_capacity_mask = token_priority <= self.expert_capacity
        expert_indices *= expert_capacity_mask
        aux_loss = self.AuxLoss(expert_indices, router_probs)
        return expert_indices, top1_probs, router_probs, aux_loss


class DenseActDense(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = nn.ReLU()

    def forward(self, hidden_states):
        hidden_states = self.wi(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.act(hidden_states)
        
        if (hidden_states.dtype != wo.weight.dtype):
            hidden_states.dtype = wo.weight.dtype
        hidden_states = self.wo(hidden_states)

        return hidden_states

class Experts(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.n_experts = config.n_experts
        self.experts = nn.ModuleDict()
        for i in range(self.n_experts):
            self.experts[f'expert_{i}'] = DenseActDense(config)

    def forward(self, hidden_states, selected_experts, routing_weights):
        final_hidden_states = torch.zeros_like(hidden_states)
        expert_mask = selected_experts.permute(2, 1, 0)

        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
            current_state = hidden_states[None, top_x].reshape(-1, hidden_states.shape[-1])
            current_hidden_states = self.experts[f"expert_{expert_idx[0]}"](current_state) * routing_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        return final_hidden_states



class SparseMLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.router = Router(config)
        self.experts = Experts(config)

    def forward(self, hidden_states):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        _, selected_experts, routing_weights = self.router(hidden_states)
        hidden_states = self.experts(hidden_states, selected_experts, routing_weights)
        hidden_states = hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return hidden_states

    
class LayerFF(nn.Module):
    def __init__(self, config, is_sparse = False):
        super().__init__()
        if is_sparse:
            self.ff = SparseMLP(config)
        else:
            self.ff = DenseActDense(config)

    def forward(self, hidden_states):
        hidden_states = self.ff(hidden_states)
        return hidden_states
    

class Attention(nn.Module):
    def __init__(self, config, masked = True):
        assert config.d_kv%config.n_heads == 0
        super().__init__()
        self.masked = masked
        self.query = nn.Linear(config.d_model, config.d_kv, bias = False)
        self.key = nn.Linear(config.d_model, config.d_kv, bias = False)
        self.value = nn.Linear(config.d_model, config.d_kv, bias = False)
        self.o = nn.Linaer(config.d_kv, config.d_model)
    
    def forward(self, hidden_states,encoder_states = None):
        B,seq_len,d_model = hidden_states.shape
        q = self.query(hidden_states).view(B,config.n_heads, seq_len, config.d_kv//config.n_heads)

        if not encoder_states:
            encoder_states = hidden_states

        k = self.key(encoder_states).view(B,config.n_heads, seq_len, config.d_kv//config.n_heads)
        v = self.value(encoder_states).view(B,config.n_heads, seq_len, config.d_kv//config.n_heads)

        wei = q@k.transpose(-1,-2)

        if masked:
            mask = torch.tril(torch.ones(seq_len,seq_len))
            wei = torch.masked_fill(wei, mask == 0, -torch.inf)

        attn = F.softmax(wei/(config.d_kv//n_heads), 3) @ v
        out = attn.transpose(1,2).view(B, seq_len, d_model)
        out = self.o(out)

        return out
        
class SelfAttention(nn.Module):
    def __init__(self, config, masked = True):
        super().__init__()
        self.selfattn = Attention(config, masked)

    def forward(self, hidden_states):
        return self.selfattn(hidden_states)

class CrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.enc_dec_attn = Attention(config, masked=False)

    def forward(self, hidden_states, encoder_states):
        return self.enc_dec_attn(hidden_states, encoder_states)


class EncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.selfattn1 = SelfAttention(config, masked=False)
        self.spareMLP = LayerFF(config,is_sparse=True)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.selfattn2 = SelfAttention(config, masked=False)
        self.denseMLP = LayerFF(config, is_sparse=False)

    def forward(self, hidden_states):
        hidden_states = self.ln1(hidden_states)
        hidden_states = hidden_states + self.selfattn1(hidden_states)  #residual connection
        hidden_states = self.spareMLP(hidden_states)
        hidden_states = self.ln2(hidden_states) 
        hidden_states = hidden_states + self.selfattn2(hidden_states)  # residual connection
        hidden_states = self.denseMLP(hidden_states)
        return hidden_states

class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.selfattn = SelfAttention(config, masked=True)
        self.sparseMLP = LayerFF(config, is_sparse=True)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.crossattn = CrossAttention(config)
        self.denseMLP = LayerFF(config, is_sparse=True)

    def forward(self, hidden_states,encoder_states):
        hidden_states = self.ln1(hidden_states)
        hidden_states = hidden_states + self.selfattn(hidden_states)  #residual connection
        hidden_states = self.sparseMLP(hidden_states)
        hidden_states = self.ln2(hidden_states)
        hidden_states = hidden_states + self.crossattn(hidden_states,encoder_states)  #residaul connection
        hidden_states = self.denseMLP(hidden_states)
        return hidden_states


class SwitchTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        if config.is_decoder:
            Block = DecoderBlock 
        else:
            Block = EncoderBlock

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.d_model),
            wpe = nn.Embedding(config.block_size, d_model),
            h = nn.ModuleList([ Block(config) for _ in range(config.n_layers)])
        ))
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)

        self.lm_head.weight = self.transformer.wte.weight  # weight sharing

        self.apply(self._init_weights)

    def _init_weights(self, module):
        mean, std = 0, 0.02
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight,mean, std)
            nn.init.zeros_(module)
        if isinstance(module, Embedding):
            nn.init.normal_(module,mean, std)

    def forward(self, idx,  encoder_states=None,targets = None):
        inp_embs = self.transformer.wte(idx)
        pos_embs = self.transformer.wpe(idx)

        x = inp_embs + pos_embs

        for block in self.transformer.h:
            x = block(x, encoder_states)
        out = self.lm_head(x)

        loss = F.cross_entropy(out, targets)
        return out,loss
    

class AuxLoss(nn.Module):
    def __init__(self, config):
        self.seq_len = config.seq_len
        self.n_experts = config.n_experts
    
    def forward(self, expert_indices, router_probs):
        fi = expert_indices.sum(-2) / self.seq_len #(batch_size, n_experts)
        pi = router_probs.sum(-2) / self.seq_len   #(batch_size, n_experts)
        aux_loss = self.n_experts* (fi*pi).sum(-1) #(batch_size,)
        return aux_loss.mean()



class SwitchTransformerConfig():
    """
    Args:
    d_model: hidden state dimention
    d_kv: key
    """
    pass    






