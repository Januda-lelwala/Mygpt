import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken


# hyperparameters
batch_size = 4#64 # how many independent sequences will we process in parallel?
block_size = 8#256 # what is the maximum context length for predictions?
max_iters = 1000
eval_interval = 100
learning_rate = 3e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device ='mps' if torch.backends.mps.is_available() else 'cpu'
eval_iters = 200
n_embd = 30#384
n_head = 3#6
n_layer = 6
dropout = 0.2







with open('alllines.txt','r',encoding='utf-8') as f:
    text=f.read()

tokenizer=tiktoken.get_encoding("cl100k_base")
vocab_size=tokenizer.n_vocab
tokens=tokenizer.encode_ordinary(text)
data=torch.tensor(tokens,dtype=torch.long)

n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class multihead_attention(nn.Module):
    def __init__(self,embed_dims,num_heads):
        super().__init__()
        assert embed_dims%num_heads==0

        self.embed_dims=embed_dims
        self.num_heads=num_heads
        self.dim_per_head=embed_dims//num_heads
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.q=nn.Linear(embed_dims,embed_dims)
        self.k=nn.Linear(embed_dims,embed_dims)
        self.v=nn.Linear(embed_dims,embed_dims)

        self.final_linear=nn.Linear(embed_dims,embed_dims)
        self.dropout=nn.Dropout(p=0.1)

    def forward(self,x):

        batch_size,T,C=x.shape

        Q=self.q(x).view(batch_size,-1,self.num_heads,self.dim_per_head).transpose(1,2)
        K=self.k(x).view(batch_size,-1,self.num_heads,self.dim_per_head).transpose(1,2)
        V=self.v(x).view(batch_size,-1,self.num_heads,self.dim_per_head).transpose(1,2)

        scores=(Q @ K.transpose(-2,-1))/(self.dim_per_head**0.5)
        scores=scores.masked_fill(self.tril[:T,:T]==0,float('-inf'))
        attn=F.softmax(scores,dim=-1)
        attn=self.dropout(attn)
        wei=attn @ V

        out=wei.transpose(1,2).contiguous().view(batch_size,-1,self.embed_dims)
        out=self.final_linear(out)

        return out

class feedforward(nn.Module):
    def __init__(self,embed_dims):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self,x):
        return self.net(x)

class block(nn.Module):
    def __init__(self,embed_dims,num_heads):
        super().__init__()
        #slef.num_heads=num_heads
        self.mha=multihead_attention(embed_dims,num_heads)
        self.feedforward=feedforward(embed_dims)
        self.ln1=nn.LayerNorm(embed_dims)
        self.ln2=nn.LayerNorm(embed_dims)

    def forward(self,x):
        
        x=x+self.mha(self.ln1(x))
        x=x+self.feedforward(self.ln2(x))

        return x

class gpt(nn.Module):
     def __init__(self):
        super().__init__()
        self.token_embeddings=nn.Embedding(vocab_size,n_embd)
        self.positional_encodings=nn.Embedding(vocab_size,n_embd)
        self.blocks=nn.Sequential(*[block(n_embd,n_head) for _ in range(n_layer)])
        self.ln_f=nn.LayerNorm(n_embd)
        self.lm_head=nn.Linear(n_embd,vocab_size)
        self.apply(self._init_weights)

     def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
     def forward(self,idx,targets=None):
        B,T=idx.shape
        
        
        tok_emb=self.token_embeddings(idx)
        pos_enc=self.positional_encodings(idx)
        x=tok_emb+pos_enc
        x=self.blocks(x)
        logits=self.lm_head(self.ln_f(x))
        if targets==None:
            loss=None
        else:
            B,T,C=logits.shape
            logits=logits.view(B*T,C)
            targets=targets.view(B*T)
            loss=F.cross_entropy(logits,targets)
        
        return logits,loss

     def generate(self,idx,max_tokens):

        for _ in range(max_tokens):
            idx=idx[:,-block_size:]
            logits,loss=self(idx)
            logits=logits[:,-1,:]
            probs=F.softmax(logits,dim=-1)
            
            idx_next=torch.multinomial(probs,num_samples=1)
            idx=torch.cat((idx,idx_next),dim=1)
        return idx


model=gpt()
m=model.to(device)

print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
            

        

        
        
        
            

        



