"""
Interactive Chat Interface for TinyGPT

A simple command-line interface to chat with your trained TinyGPT model.
"""

import torch
import sentencepiece as spm
import torch.nn.functional as F
from tiny_gpt import TinyGPT, Config

def top_k_top_p_sampling(logits, top_k=50, top_p=0.9, temperature=1.0):
    """Apply top-k and top-p filtering to logits for more diverse generation."""
    
    if temperature != 1.0:
        logits = logits / temperature
    
    # Top-k filtering
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')
    
    # Top-p filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')
    
    return logits

def generate_response(model, tokenizer, prompt, config, device, 
                     max_length=50, top_k=50, top_p=0.9, temperature=0.8):
    """
    Generate a response using advanced sampling techniques.
    
    Args:
        model: Trained TinyGPT model
        tokenizer: SentencePiece tokenizer
        prompt: Input text prompt
        config: Model configuration
        device: torch device
        max_length: Maximum tokens to generate
        top_k: Top-k sampling parameter
        top_p: Top-p (nucleus) sampling parameter
        temperature: Temperature for softmax sampling
    """
    
    model.eval()
    
    # Tokenize input
    tokens = tokenizer.encode(prompt, out_type=int)
    
    # Ensure we have space for generation
    if len(tokens) >= config.seq_len - max_length:
        tokens = tokens[-(config.seq_len - max_length - 1):]
    
    input_tensor = torch.tensor([tokens], dtype=torch.long).to(device)
    
    generated_tokens = []
    
    with torch.no_grad():
        for step in range(max_length):
            if input_tensor.size(1) >= config.seq_len:
                break
            
            # Get model predictions
            log_probs = model(input_tensor)
            next_token_logits = log_probs[:, -1, :].clone()
            
            # Apply sampling
            filtered_logits = top_k_top_p_sampling(
                next_token_logits, top_k=top_k, top_p=top_p, temperature=temperature
            )
            
            # Sample from the filtered distribution
            probs = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Stop if EOS token is generated
            if next_token.item() == tokenizer.piece_to_id("</s>"):
                break
            
            # Add to sequence
            input_tensor = torch.cat([input_tensor, next_token], dim=1)
            generated_tokens.append(next_token.item())
    
    # Decode only the newly generated part
    if generated_tokens:
        response = tokenizer.decode(generated_tokens)
        return response.strip()
    else:
        return ""

def load_model_and_tokenizer(model_path="best_model.pt", tokenizer_path="chatbot_tokenizer.model"):
    """Load the trained model and tokenizer."""
    
    # Load tokenizer
    tokenizer = spm.SentencePieceProcessor()
    try:
        tokenizer.load(tokenizer_path)
    except FileNotFoundError:
        print(f"❌ Tokenizer file '{tokenizer_path}' not found!")
        print("Make sure you've trained the model first by running: python tiny_gpt.py")
        return None, None
    
    # Initialize model with correct vocab size
    config = Config()
    config.vocab_size = tokenizer.get_piece_size()
    model = TinyGPT(config)
    
    # Load model weights
    try:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print(f"✅ Model loaded from {model_path}")
    except FileNotFoundError:
        print(f"❌ Model file '{model_path}' not found!")
        print("Make sure you've trained the model first by running: python tiny_gpt.py")
        return None, None
    
    return model, tokenizer

def interactive_chat():
    """Main interactive chat function."""
    
    print("🤖 TinyGPT Interactive Chat")
    print("=" * 40)
    print("Loading model and tokenizer...")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    if model is None or tokenizer is None:
        return
    
    model.to(device)
    config = Config()
    config.vocab_size = tokenizer.get_piece_size()
    
    print("\n✅ Model loaded successfully!")
    print("\n" + "=" * 50)
    print("Chat with TinyGPT! (type 'quit', 'exit', or 'bye' to stop)")
    print("Note: This model has limited training and may not always make sense.")
    print("=" * 50)
    
    # Chat loop
    while True:
        try:
            # Get user input
            user_input = input("\n👤 You: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                print("👋 Goodbye! Thanks for chatting with TinyGPT!")
                break
            
            if not user_input:
                print("Please enter a message!")
                continue
            
            # Format prompt for the model
            prompt = f"<s> User: {user_input} <sep> Bot:"
            
            print("🤖 TinyGPT: ", end="", flush=True)
            
            # Generate response
            response = generate_response(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                config=config,
                device=device,
                max_length=50,
                top_k=50,
                top_p=0.9,
                temperature=0.8
            )
            
            # Extract just the bot's response
            if "<sep> Bot:" in response:
                bot_response = response.split("<sep> Bot:")[-1].strip()
            else:
                bot_response = response.strip()
            
            # Clean up the response
            bot_response = bot_response.replace("</s>", "").strip()
            
            if bot_response:
                print(bot_response)
            else:
                print("*[No response generated]*")
        
        except KeyboardInterrupt:
            print("\n\n👋 Chat interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error generating response: {e}")
            print("Continuing chat...")

def demo_conversations():
    """Run some demo conversations to showcase the model."""
    
    print("🎬 TinyGPT Demo Conversations")
    print("=" * 40)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model_and_tokenizer()
    
    if model is None or tokenizer is None:
        return
    
    model.to(device)
    config = Config()
    config.vocab_size = tokenizer.get_piece_size()
    
    # Demo prompts
    demo_prompts = [
        "Hello! How are you today?",
        "What's your favorite color?",
        "Can you help me with something?",
        "Tell me a joke",
        "What do you like to do for fun?",
    ]
    
    print("\nGenerating sample conversations...\n")
    
    for i, user_msg in enumerate(demo_prompts, 1):
        print(f"Demo {i}:")
        print(f"👤 User: {user_msg}")
        
        prompt = f"<s> User: {user_msg} <sep> Bot:"
        
        response = generate_response(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            config=config,
            device=device,
            max_length=40,
            top_k=50,
            top_p=0.9,
            temperature=0.7
        )
        
        # Clean up response
        if "<sep> Bot:" in response:
            bot_response = response.split("<sep> Bot:")[-1].strip()
        else:
            bot_response = response.strip()
        
        bot_response = bot_response.replace("</s>", "").strip()
        
        print(f"🤖 TinyGPT: {bot_response if bot_response else '*[No response]*'}")
        print("-" * 30)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="TinyGPT Interactive Chat")
    parser.add_argument("--demo", action="store_true", 
                       help="Run demo conversations instead of interactive chat")
    parser.add_argument("--model", default="best_model.pt", 
                       help="Path to model file")
    parser.add_argument("--tokenizer", default="chatbot_tokenizer.model", 
                       help="Path to tokenizer file")
    
    args = parser.parse_args()
    
    if args.demo:
        demo_conversations()
    else:
        interactive_chat()