class SwitchTransformerConfig():
    """
    Args:
    d_model: hidden state dimention
    d_kv: key

    """
    def __init__(self,
        d_model,
        d_kv,
        d_ff,
        n_heads,
        n_experts,
        n_layers,
        router_dtype,
        capacity_factor,
        jitter_noise,
        ignore_padded_tokens,
        is_decoder,
        vocab_size,
        dropout_rate,
        weight_decay,
        seq_len,
        batch_size,
        device,
        file_path,
        optim,
        lr, 
        betas,
        eps,
        val_ratio,
        epochs,
        log_interval,

        ):

        self.d_model = d_model
        self.d_kv = d_kv
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.n_experts = n_experts      
        self.n_layers = n_layers  
        self.router_dtype = router_dtype
        self.capacity_factor = capacity_factor
        self.jitter_noise = jitter_noise
        self.ignore_padded_tokens = ignore_padded_tokens
        self.is_decoder = is_decoder
        self.vocab_size = vocab_size
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.device  = device
        self.file_path = file_path
        self.optim = optim
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.epochs = epochs
        self.val_ratio = val_ratio
        self.log_interval = log_interval


