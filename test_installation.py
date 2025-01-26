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

def _run_inference(model, tokenizer, prompt: str) -> tuple:
    """Run model inference with given prompt."""
    inference_start = time.time()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False,
        temperature=0.0,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    inference_time = time.time() - inference_start
    return response, inference_time

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
    
    try:
        if verbose:
            print("Testing installation...")
        
        # Test PyTorch environment
        env_results = _test_pytorch_environment()
        results.update(env_results)
        if verbose:
            print("\n1. Testing PyTorch:")
            for key, value in env_results.items():
                print(f"{key}: {value}")
        
        # Load model and tokenizer
        if verbose:
            print(f"\n2. Loading model and tokenizer from {model_name}")
        tokenizer, model = _load_model_and_tokenizer(model_name, device)
        if verbose:
            print("✓ Model and tokenizer loaded successfully")
        
        # Run inference
        if verbose:
            print("\n3. Testing inference:")
            print(f"Prompt: {prompt}")
        response, inference_time = _run_inference(model, tokenizer, prompt)
        
        # Collect results
        total_time = time.time() - start_time
        results.update({
            "success": True,
            "inference_time": inference_time,
            "total_time": total_time,
            "device": model.device,
            "model_name": model_name,
            "prompt": prompt,
            "response": response
        })
        
        if verbose:
            print(f"\nModel response:\n{response}")
            print(f"\n✓ Installation test completed successfully!")
            print(f"\nTiming information:")
            print(f"- Inference time: {inference_time:.2f} seconds")
            print(f"- Total test time: {total_time:.2f} seconds")
            
    except Exception as e:
        print(f"\n✗ Error during test: {str(e)}")
        results["success"] = False
        results["error"] = str(e)
    
    return results

if __name__ == "__main__":
    test_installation() 