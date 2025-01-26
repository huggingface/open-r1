from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
from tqdm import tqdm
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def _test_pytorch_environment() -> dict:
    """Test PyTorch installation and available devices."""
    results = {
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available()
    }
    return results

def _load_model_and_tokenizer(model_name: str, device: str) -> tuple:
    """Load model and tokenizer from HuggingFace."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device
    )
    return tokenizer, model

def test_installation(
    model_name: str = "Qwen/Qwen2.5-Math-1.5B-Instruct",
    device: str = "auto",
    prompt: str = "What is 2 + 2? Please reason step by step, and put your final answer within \\boxed{}.",
    verbose: bool = True
) -> dict:
    """Test ML environment installation and model inference.
    
    Args:
        model_name (str): HuggingFace model identifier
        device (str): Device to run on ('auto', 'cuda', 'cpu', 'mps')
        prompt (str): Test prompt for inference
        verbose (bool): Whether to print detailed progress
    
    Returns:
        dict: Test results including timing and device info
    """
    results = {}
    start_time = time.time()
    
    if verbose:
        print("Testing installation...")
    
    # Test PyTorch
    print("\n1. Testing PyTorch:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    
    # Test Transformers
    print("\n2. Testing Transformers:")
    print(f"Loading model and tokenizer from {model_name}")
    
    try:
        print("\nLoading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("✓ Tokenizer loaded successfully")
        
        print("\nLoading model (this may take several minutes on first run)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device
        )
        print("✓ Model loaded successfully")
        
        # Test inference
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
        
        results["success"] = True
        results["inference_time"] = inference_time
        results["total_time"] = total_time
        results["device"] = model.device
        results["model_name"] = model_name
        results["prompt"] = prompt
        results["response"] = response
        
    except Exception as e:
        print(f"\n✗ Error during test: {str(e)}")
        results["success"] = False
        results["error"] = str(e)
    
    return results

if __name__ == "__main__":
    test_installation() 