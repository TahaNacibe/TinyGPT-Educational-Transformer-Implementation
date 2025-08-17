import os
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import sentencepiece as spm
from tqdm import tqdm
import sacrebleu

# Set environment for better CUDA debugging
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

class Config:
    """Configuration class for TinyGPT model hyperparameters."""
    
    # Model architecture
    embed_dim = 384
    num_heads = 6                  
    head_dim = embed_dim // num_heads  # 64 per head
    num_layers = 8
    ff_hidden_dim = 4 * embed_dim  # 1536
    seq_len = 128
    vocab_size = 32000             
    dropout = 0.05
    
    # Training parameters
    epochs = 10
    batch_size = 32
    lr = 1e-4
    betas = (0.9, 0.95)
    weight_decay = 0.001
    clip_grad = 1.0
    early_stop_patience = 3
    
    # Special tokens
    special_tokens = ["<pad>", "<unk>", "<s>", "</s>"]

class GPTDataset(Dataset):
    """Dataset class for GPT training data."""
    
    def __init__(self, texts, tokenizer, seq_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.tokenizer.encode(self.texts[idx], out_type=int)

        # Pad or truncate to seq_len
        if len(tokens) < self.seq_len:
            tokens += [0] * (self.seq_len - len(tokens))
        else:
            tokens = tokens[:self.seq_len]

        # Create input and target sequences (shifted by 1)
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)

        return input_ids, target_ids

def causal_mask(size):
    """Create a causal (lower triangular) attention mask."""
    return torch.tril(torch.ones(size, size)).unsqueeze(0).unsqueeze(0)

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, embed_dim, num_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        B, T, E = x.shape
        
        # Project to Q, K, V and reshape for multi-head attention
        qkv = self.qkv_proj(x).chunk(3, dim=-1)
        Q, K, V = [t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) 
                   for t in qkv]

        # Compute attention scores
        scores = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Apply causal mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask[:, :, :T, :T] == 0, float("-inf"))

        # Apply softmax and compute weighted values
        attn = torch.softmax(scores, dim=-1)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, E)

        return self.out_proj(out)

class DecoderBlock(nn.Module):
    """Transformer decoder block with self-attention and feed-forward layers."""
    
    def __init__(self, embed_dim, num_heads, head_dim, ff_hidden_dim, dropout=0.1):
        super().__init__()
        
        self.attn = MultiHeadAttention(embed_dim, num_heads, head_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim, embed_dim)
        )

    def forward(self, x, mask):
        # Self-attention with residual connection
        x = x + self.dropout1(self.attn(self.norm1(x), mask))
        # Feed-forward with residual connection
        x = x + self.dropout2(self.ff(self.norm2(x)))
        return x

class TinyGPT(nn.Module):
    """
    TinyGPT: A minimal GPT implementation for educational purposes.
    
    Note: This model is trained on limited data with few epochs and layers.
    It can generate text but may not always make coherent sense due to
    resource limitations. This is expected behavior for this educational implementation.
    """
    
    def __init__(self, config):
        super().__init__()
        
        # Token and positional embeddings
        self.token_embd = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_embd = nn.Embedding(config.seq_len, config.embed_dim)
        
        # Transformer decoder blocks
        self.blocks = nn.ModuleList([
            DecoderBlock(config.embed_dim, config.num_heads, config.head_dim, 
                        config.ff_hidden_dim, config.dropout) 
            for _ in range(config.num_layers)
        ])
        
        # Final layer norm and output projection
        self.ln = nn.LayerNorm(config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)
        
        # Using adaptive softmax for efficiency with large vocabulary
        self.head = nn.AdaptiveLogSoftmaxWithLoss(
            in_features=config.embed_dim,
            n_classes=config.vocab_size,
            cutoffs=[2000, 10000],
            div_value=4.0,
            head_bias=False
        )
        
        # Tie embedding weights (common practice in language models)
        self.head.weight = self.token_embd.weight

    def forward(self, x, target=None):
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0)

        # Apply embeddings
        x = self.token_embd(x) + self.pos_embd(positions)
        x = self.dropout(x)

        # Apply causal mask and pass through transformer blocks
        mask = causal_mask(T).to(x.device)
        for block in self.blocks:
            x = block(x, mask)

        # Final layer normalization
        x = self.ln(x)
        x = x.view(-1, x.size(-1))  # Flatten for adaptive softmax

        if target is not None:
            # Training mode: compute loss
            target = target.view(-1)
            output = self.head(x, target)
            return output
        else:
            # Inference mode: return log probabilities
            log_probs = self.head.log_prob(x)
            return log_probs.view(B, T, -1)

def prepare_data():
    """Load and prepare training data from prosocial-dialog and daily_dialog datasets."""
    
    print("Loading datasets...")
    
    # Load prosocial dialog dataset
    prosocial_train = load_dataset("allenai/prosocial-dialog", split="train")
    prosocial_texts_train = [
        f"<s> User: {d['context']} <sep> Bot: {d['response']} </s>" 
        for d in prosocial_train
    ]
    
    prosocial_test = load_dataset("allenai/prosocial-dialog", split="test")
    prosocial_texts_test = [
        f"<s> User: {d['context']} <sep> Bot: {d['response']} </s>" 
        for d in prosocial_test
    ]
    
    prosocial_val = load_dataset("allenai/prosocial-dialog", split="validation")
    prosocial_texts_val = [
        f"<s> User: {d['context']} <sep> Bot: {d['response']} </s>" 
        for d in prosocial_val
    ]
    
    # Load daily dialog dataset
    daily_train = load_dataset("daily_dialog", split="train")
    daily_texts_train = [
        f"<s> User: {d['dialog'][0]} <sep> Bot: {d['dialog'][1]} </s>" 
        for d in daily_train if len(d["dialog"]) >= 2
    ]
    
    daily_test = load_dataset("daily_dialog", split="test")
    daily_texts_test = [
        f"<s> User: {d['dialog'][0]} <sep> Bot: {d['dialog'][1]} </s>" 
        for d in daily_test if len(d["dialog"]) >= 2
    ]
    
    daily_val = load_dataset("daily_dialog", split="validation")
    daily_texts_val = [
        f"<s> User: {d['dialog'][0]} <sep> Bot: {d['dialog'][1]} </s>" 
        for d in daily_val if len(d["dialog"]) >= 2
    ]
    
    # Combine datasets
    train_texts = prosocial_texts_train + daily_texts_train
    test_texts = prosocial_texts_test + daily_texts_test
    val_texts = prosocial_texts_val + daily_texts_val
    
    random.shuffle(train_texts)
    
    print(f"Prepared {len(train_texts)} training samples")
    print(f"Prepared {len(test_texts)} test samples")
    print(f"Prepared {len(val_texts)} validation samples")
    
    return train_texts, test_texts, val_texts

def train_tokenizer(texts, config):
    """Train a SentencePiece tokenizer on the provided texts."""
    
    print("Training tokenizer...")
    
    # Save texts to file for SentencePiece training
    with open("train_text.txt", "w", encoding="utf-8") as f:
        for line in texts:
            f.write(line.strip() + "\n")
    
    # Train SentencePiece tokenizer
    spm.SentencePieceTrainer.train(
        input='train_text.txt',
        model_prefix='chatbot_tokenizer',
        vocab_size=config.vocab_size,
        model_type='bpe',
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        user_defined_symbols=['<sep>', 'User:', 'Bot:', '<s>', '</s>']
    )
    
    # Load the trained tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load("chatbot_tokenizer.model")
    
    print(f"Tokenizer trained with vocabulary size: {sp.get_piece_size()}")
    
    return sp

def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    """Create a learning rate scheduler with cosine annealing and warmup."""
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)

def compute_bleu_score(predicted_ids, target_ids, tokenizer):
    """Compute BLEU score for evaluation."""
    
    predictions = []
    references = []

    for pred, true in zip(predicted_ids, target_ids):
        pred_tokens = pred.tolist()
        true_tokens = true.tolist()

        # Remove padding tokens
        if 0 in pred_tokens:
            pred_tokens = pred_tokens[:pred_tokens.index(0)]
        if 0 in true_tokens:
            true_tokens = true_tokens[:true_tokens.index(0)]

        # Decode to text
        pred_text = tokenizer.decode(pred_tokens)
        true_text = tokenizer.decode(true_tokens)

        predictions.append(pred_text)
        references.append(true_text)

    bleu = sacrebleu.corpus_bleu(predictions, [references])
    return bleu.score

def evaluate_model(model, dataloader, device, tokenizer):
    """Evaluate the model on a dataset."""
    
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for input_ids, target_ids in tqdm(dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            # Compute loss
            output = model(input_ids, target_ids)
            loss = output.loss
            total_loss += loss.item()

            # Get predictions for BLEU score
            log_probs = model(input_ids)
            predicted_ids = torch.argmax(log_probs, dim=-1)

            all_preds.extend(predicted_ids.cpu())
            all_targets.extend(target_ids.cpu())

    avg_loss = total_loss / len(dataloader)
    bleu_score = compute_bleu_score(all_preds, all_targets, tokenizer)

    return avg_loss, bleu_score

def train_model(model, train_loader, val_loader, config, device, tokenizer):
    """Train the TinyGPT model."""
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.lr, 
        betas=config.betas, 
        weight_decay=config.weight_decay
    )
    
    total_steps = config.epochs * len(train_loader)
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    model = model.to(device)
    best_val_loss = float("inf")
    patience = 0

    for epoch in range(1, config.epochs + 1):
        print(f"\n=== Epoch {epoch}/{config.epochs} ===")
        
        # Training phase
        model.train()
        total_loss = 0

        for input_ids, target_ids in tqdm(train_loader, desc=f"Training"):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            optimizer.zero_grad()
            output = model(input_ids, target_ids)
            loss = output.loss
            loss.backward()

            if config.clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)

            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_ppl = math.exp(avg_train_loss)

        # Evaluation phase
        val_loss, val_bleu = evaluate_model(model, val_loader, device, tokenizer)
        val_ppl = math.exp(val_loss)

        print(f"Train Loss: {avg_train_loss:.4f}, Train PPL: {train_ppl:.2f}")
        print(f"Val Loss: {val_loss:.4f}, Val PPL: {val_ppl:.2f}, Val BLEU: {val_bleu:.2f}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
            torch.save(model.state_dict(), "best_model.pt")
            print("✅ New best model saved!")
        else:
            patience += 1
            print(f"Patience: {patience}/{config.early_stop_patience}")
            if patience >= config.early_stop_patience:
                print("Early stopping triggered!")
                break

def simple_generate(model, tokenizer, prompt, config, device, max_length=50):
    """Generate text using the trained model."""
    
    model.eval()
    
    # Tokenize input
    tokens = tokenizer.encode(prompt, out_type=int)
    tokens = tokens[:config.seq_len - max_length]  # Leave space for generation
    
    input_tensor = torch.tensor([tokens], dtype=torch.long).to(device)
    
    with torch.no_grad():
        for _ in range(max_length):
            if input_tensor.size(1) >= config.seq_len:
                break
                
            # Get model predictions
            log_probs = model(input_tensor)
            next_token_logits = log_probs[:, -1, :]
            
            # Simple greedy sampling
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Stop if EOS token is generated
            if next_token.item() == tokenizer.piece_to_id("</s>"):
                break
                
            input_tensor = torch.cat([input_tensor, next_token], dim=1)
    
    # Decode the generated sequence
    generated_tokens = input_tensor[0].tolist()
    response = tokenizer.decode(generated_tokens)
    
    return response

def main():
    """Main training and evaluation function."""
    
    # Configuration
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Prepare data
    train_texts, test_texts, val_texts = prepare_data()
    
    # Train tokenizer
    tokenizer = train_tokenizer(train_texts, config)
    config.vocab_size = tokenizer.get_piece_size()  # Update vocab size
    
    # Create datasets and dataloaders
    train_dataset = GPTDataset(train_texts, tokenizer, config.seq_len)
    test_dataset = GPTDataset(test_texts, tokenizer, config.seq_len)
    val_dataset = GPTDataset(val_texts, tokenizer, config.seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Initialize model
    model = TinyGPT(config)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Train model
    print("\n🚀 Starting training...")
    train_model(model, train_loader, val_loader, config, device, tokenizer)
    
    # Load best model and evaluate
    model.load_state_dict(torch.load("best_model.pt"))
    test_loss, test_bleu = evaluate_model(model, test_loader, device, tokenizer)
    
    print(f"\n📊 Final Test Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Perplexity: {math.exp(test_loss):.2f}")
    print(f"Test BLEU: {test_bleu:.2f}")
    
    # Demo generation
    print("\n🤖 Demo Generation:")
    prompt = "<s> User: Hello! How are you? <sep> Bot:"
    response = simple_generate(model, tokenizer, prompt, config, device)
    print(f"Input: {prompt}")
    print(f"Generated: {response}")
    
    print("\n✅ Training completed!")

if __name__ == "__main__":
    main()