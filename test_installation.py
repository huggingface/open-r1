from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
from tqdm import tqdm
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def test_installation():
    print("Testing installation...")
    start_time = time.time()
    
    # Test PyTorch
    print("\n1. Testing PyTorch:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    
    # Test Transformers
    print("\n2. Testing Transformers:")
    model_name = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    print(f"Loading model and tokenizer from {model_name}")
    
    try:
        print("\nLoading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("✓ Tokenizer loaded successfully")
        
        print("\nLoading model (this may take several minutes on first run)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print("✓ Model loaded successfully")
        
        # Test inference
        prompt = "What is 2 + 2? Please reason step by step, and put your final answer within \\boxed{}."
        print("\n3. Testing inference:")
        print(f"Prompt: {prompt}")
        
        print("\nGenerating response (this may take a minute on first run)...")
        inference_start = time.time()
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        print(f"Input processing complete. Device: {model.device}")
        
        # Reduced complexity generation
        max_tokens = 50  # Reduced from 100
        print(f"\nGenerating up to {max_tokens} tokens with simplified settings...")
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,  # Deterministic generation
            temperature=0.0,  # Remove randomness
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        inference_time = time.time() - inference_start
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nModel response:\n{response}")
        
        total_time = time.time() - start_time
        print(f"\n✓ Installation test completed successfully!")
        print(f"\nTiming information:")
        print(f"- Inference time: {inference_time:.2f} seconds")
        print(f"- Total test time: {total_time:.2f} seconds")
        
    except Exception as e:
        print(f"\n✗ Error during test: {str(e)}")

if __name__ == "__main__":
    test_installation() 