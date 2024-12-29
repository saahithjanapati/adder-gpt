import torch
import torch.nn.functional as F
import torch.nn as nn
import math



itos = {
    0: "0",
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "+",
    11: "=",
    12: "$",
    13: "."
}

stoi = {v: k for k,v in itos.items()}



class LayerNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.my_weight = nn.Parameter(torch.ones(config.n_embd))
        self.my_bias = nn.Parameter(torch.zeros(config.n_embd)) if config.use_bias else None
        
    
    def forward(self, x):
        return F.layer_norm(x, 
                     normalized_shape=self.my_weight.shape, 
                     weight=self.my_weight, 
                     bias=self.my_bias, 
                     eps=1e-5)


class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.use_bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.use_bias)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        self.head_size = int(config.n_embd / config.n_head) # h_dim
        self.num_head = config.n_head # number of heads
        self.block_size = config.block_size
        
        self.register_buffer("att_mask", torch.triu(float('-inf') * torch.ones(config.block_size, config.block_size), diagonal=1).view(1, 1, self.block_size, self.block_size))
        
        
    def forward(self, x):
        b, t, c = x.size()
        
        q, k, v = self.c_attn(x).split(c, dim=-1) # (b,t,c)
        
        q = q.view(b, t, self.num_head, self.head_size).transpose(2,1) # (b, n_h, t, h_dim)
        k = k.view(b, t, self.num_head, self.head_size).transpose(2,1) # (b, n_h, t, h_dim)
        v = v.view(b, t, self.num_head, self.head_size).transpose(2,1) # (b, n_h, t, h_dim)
        
        att_scores = q @ k.transpose(-1, -2) / math.sqrt(self.head_size) # (b, n_h, t, h_dim) @ (b, n_h, h_dim, t) --> (b, n_h, t, t)
        
        # mask the scores
        att_scores += self.att_mask[:, :, :t, :t]
        
        att_scores = torch.softmax(att_scores, dim=-1) # perform softmax for each element
        
        # perform dropout
        att_scores = self.attn_dropout(att_scores) #(b, n_h, t, t)
        
        out = att_scores @ v # (b, n_h, t, t) @ (b, b_h, t, h_dim) --> (b, n_h, t, h_dim)

        out = out.transpose(1, 2).contiguous() # (b, t, n_h, h_dim)
        out = out.view(b, t, c)
        out = self.resid_dropout(self.c_proj(out))
        
        return out

    
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.up_proj = nn.Linear(config.n_embd, 4*config.n_embd, bias=config.use_bias)
        self.dropout = nn.Dropout(config.dropout)
        self.gelu = nn.GELU()
        self.down_proj = nn.Linear(config.n_embd * 4, config.n_embd, bias=config.use_bias)
        
        
    def forward(self, x):
        out = self.gelu(self.up_proj(x))
        out = self.down_proj(self.dropout(out))
        return out


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.ln_1 = LayerNorm(config)
        self.sa = SelfAttention(config)
        
        self.ln_2 = LayerNorm(config)
        self.mlp = MLP(config)
    
    
    def forward(self, x):
        x = x +  self.sa(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        
        self.layers = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        
        self.ln_f = LayerNorm(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=config.use_bias)
        # self.lm_head.weight = self.wte.weight
    
    
    def forward(self, idx, targets=None):
        # make sure x is on same device
        device = next(self.parameters()).device
        
        idx = idx.to(device)
        
        b, t = idx.size()
        
        pos = torch.arange(0, t, device=device, dtype=torch.long)
        
        tok_emb = self.wte(idx) # (b, t, c)
        pos_emb = self.wpe(pos) # (t,) --> (t, c)
        
        x = tok_emb + pos_emb # (b, t, c)
        
        x = self.drop(x)
                
        for layer in self.layers:
            x = layer(x)
            
        
        logits = self.lm_head(x) # (b,t, c) --> (b, t, vocab_size)
        
        loss = None
        # compute cross entropy loss if targets are provided
        if targets is not None: # targets will be provided in shape b, t
            unrolled_targets = targets.view(b*t)
            unrolled_logits = logits.view(b*t, -1)
            
            loss = F.cross_entropy(unrolled_logits, unrolled_targets, ignore_index=-1)        
                
        return logits, loss



def generate(idx, model):
    logits = None
    while idx[0][-1] != stoi["$"]:
        logits, _ = model.forward(idx)
        next_idx = torch.tensor([[logits[0][-1].argmax().item()]])
        idx = torch.cat([idx, next_idx], dim=-1)

    return idx[0][:-1], logits

        
def get_best_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')



class GPTConfig:
    vocab_size = 14
    block_size = 22
    n_embd = 64
    n_head = 8
    n_layer = 6
    use_bias = True
    dropout = 0.1


def main():
    # load the model

    best_device = get_best_device()

    state_dict = torch.load("adder_gpt_state_dict.pth", map_location = best_device)
    model = GPT(GPTConfig())

    model.load_state_dict(state_dict)
    
    # read user input
    a = int(input("Input a positive integer a, 6 digits or less: ").strip())
    b = int(input("Input a positive integer b, 6 digits or less: ").strip())

    correct_answer = a + b

    input_string = f"{a}+{b}="

    full_string = input_string + str(correct_answer)[::-1] + "$"
    full_labels = torch.tensor([stoi[c] for c in full_string])[1:]
    full_labels[:len(input_string)-1] = -1

    input_tensor = torch.tensor([[stoi[c] for c in input_string]]).long()

    # generate the answer
    generated, logits = generate(input_tensor, model)

    if logits.size(1) != full_labels.size(0):
        print("Output size does not match, skipping loss calculation")
        loss = None
    
    else:
        loss = F.cross_entropy(logits[0], full_labels, ignore_index=-1).item()

    # decode generated answer
    generated_indices = generated.tolist()
    split_idx = generated_indices.index(11)
    output = int("".join(list(map(str, generated_indices[split_idx+1:][::-1]))))
    
    if output == correct_answer:
        print(f"Generated answer: {output} \u2705")  # Green check mark
    else:
        print(f"Generated answer: {output} \u274C")  # Red cross mark

    print(f"Loss: {loss}")
    





if  __name__ == "__main__":
    main()