# TinyGPT 

A minimal GPT implementation for educational purposes and experimentation with limited computational resources.

## Important Notice

**This model is trained on a small dataset with limited epochs and few hidden layers. It can generate text but may not always produce coherent or sensible responses. This is expected behavior given the resource constraints and is intended for educational purposes only.**

## Purpose

TinyGPT is designed for:
- Learning transformer architecture fundamentals
- Experimentation with limited GPU memory (15GB VRAM friendly)
- Understanding GPT training pipeline
- Educational demonstrations

## Architecture

- **Embedding Dimension**: 384
- **Number of Heads**: 6 (64 dimensions per head)
- **Number of Layers**: 8
- **Feed-Forward Hidden**: 1536
- **Sequence Length**: 128 tokens
- **Vocabulary Size**: ~32,000 tokens
- **Parameters**: ~50M (approximate)

## Training Configuration

- **Datasets**: 
  - ProSocial Dialog (Allen AI)
  - Daily Dialog
- **Epochs**: 10 (limited for resource efficiency)
- **Batch Size**: 32
- **Learning Rate**: 1e-4 with cosine scheduling
- **Dropout**: 0.05
- **Gradient Clipping**: 1.0

## Quick Start

### Installation

```bash
git clone https://github.com/yourusername/TinyGPT.git
cd TinyGPT
pip install -r requirements.txt
```

### Training

```bash
python tiny_gpt.py
```

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended, 8GB+ VRAM)
- 16GB+ RAM

## Usage Example

```python
from tiny_gpt import TinyGPT, Config
import sentencepiece as spm

# Load trained model and tokenizer
config = Config()
model = TinyGPT(config)
model.load_state_dict(torch.load("best_model.pt"))

tokenizer = spm.SentencePieceProcessor()
tokenizer.load("chatbot_tokenizer.model")

# Generate response
prompt = "<s> User: Hello! <sep> Bot:"
response = simple_generate(model, tokenizer, prompt, config, device)
print(response)
```

## File Structure

```
TinyGPT/
├── tiny_gpt.py              # Main implementation
├── requirements.txt         # Dependencies
├── README.md               # This file
├── LICENSE                 # MIT License
├── .gitignore             # Git ignore rules
└── examples/              # Usage examples
    └── interactive_chat.py # Simple chat interface
```

## Model Components

### MultiHeadAttention
- Implements scaled dot-product attention
- Causal masking for autoregressive generation
- 6 attention heads with 64 dimensions each

### DecoderBlock
- Pre-normalization architecture
- GELU activation function
- Residual connections

### TinyGPT
- Token and positional embeddings
- 8 transformer decoder layers
- Adaptive softmax for efficient vocabulary handling

## Training Process

1. **Data Loading**: Combines ProSocial Dialog and Daily Dialog datasets
2. **Tokenization**: Trains custom SentencePiece BPE tokenizer
3. **Training Loop**: 
   - AdamW optimizer with weight decay
   - Cosine learning rate scheduling with warmup
   - Gradient clipping
   - Early stopping based on validation loss
4. **Evaluation**: BLEU score and perplexity metrics

## Performance Notes

**Expected Behavior:**
- ✅ Can generate grammatically structured text
- ✅ Follows basic conversational patterns
- ❌ Will produce nonsensical or repetitive responses
- ❌ Limited understanding of context and semantics
- ❌ Not suitable for production use
**Note** i'm really running that on colab ree tear so it's expected

This is **intentional** due to:
- Small training dataset
- Limited training epochs (10)
- Compact model size (8 layers, 384 dimensions)
- Resource constraints

## Customization

### Adjusting Model Size
```python
# In Config class
embed_dim = 512        # Increase embedding dimension
num_layers = 12        # Add more layers
num_heads = 8          # More attention heads
```

### Training Parameters
```python
# In Config class
epochs = 20            # More training epochs
batch_size = 64        # Larger batches (requires more VRAM)
lr = 5e-5             # Different learning rate
```

## Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size` in Config
- Decrease `seq_len` or `embed_dim`
- Use gradient checkpointing (not implemented in basic version)

### Poor Generation Quality
- Increase training epochs
- Use larger model dimensions
- Add more training data
- Implement better sampling strategies

### Tokenizer Issues
- Ensure all special tokens are properly defined
- Check vocabulary size matches model configuration

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! This is an educational project aimed at helping others learn transformer architectures.

### Areas for Improvement
- Better sampling strategies (top-k, top-p, temperature)
- Gradient checkpointing for memory efficiency
- Data augmentation techniques
- Model parallelism
- Better evaluation metrics

## Educational Resources

This implementation demonstrates:
- Transformer decoder architecture
- Self-attention mechanisms
- Positional encodings
- Causal language modeling
- Training loop best practices
- Tokenization with SentencePiece

## Acknowledgments

- Inspired by GPT papers from OpenAI
- Built using PyTorch and Hugging Face datasets
- SentencePiece for tokenization
- Educational purpose implementation

---

**Remember**: This is a learning tool, not a production model. The limited coherence is expected and designed to demonstrate transformer fundamentals within resource constraints.
