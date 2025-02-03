from datasets import Dataset
from open_r1.generate import build_distilabel_pipeline

# 1. First, make sure you have a SGLang server running:
# ```bash
# # Example of starting SGLang server (adjust according to your setup)
# sglang start-server --model meta-llama/Llama-2-7b-chat-hf --port 8000
# ```
# Then run the test script:
# ```bash
# python test_sglang_backend.py
# ```

# Some key points about the implementation:
# The test uses a tiny dataset with just 2 prompts to verify functionality quickly
# We're using small values for batch size and max_new_tokens to keep resource usage low
# The timeout is reduced to 30 seconds for faster testing
# Need to adjust the model name and server URL according to your setup
# To verify different aspects, we may:
# - Test error handling:
# - Test different generation parameters:
# - Test batch processing:

# The main differences between vLLM and SGLang backends in your current implementation are:
# The API interface (OpenAI-compatible vs SGLang native)
# Parameter handling (generation_kwargs are passed differently)
# If you encounter any issues, the most likely points of failure would be:
# API compatibility
# Parameter mapping between backends
# Response format differences


def test_sglang_backend():
    # Create a tiny test dataset
    test_data = {
        "prompt": [
            "What is 2+2?",
            "Write a one-sentence story.",
        ]
    }
    test_dataset = Dataset.from_dict(test_data)

    # Initialize the pipeline with SGLang backend
    pipeline = build_distilabel_pipeline(
        model="meta-llama/Llama-2-7b-chat-hf",  # Replace with your model
        backend="sglang",
        base_url="http://localhost:8000",  # Adjust to your SGLang server URL
        prompt_column="prompt",
        temperature=0.7,
        max_new_tokens=100,  # Smaller for testing
        num_generations=1,
        input_batch_size=2,  # Small batch size for testing
        timeout=30,  # Shorter timeout for testing
    )

    # Run inference
    print("Running inference...")
    results = pipeline.run(
        dataset=test_dataset,
        dataset_batch_size=2,
        use_cache=False,
    )
    
    # Print results
    print("\nGenerated responses:")
    for i, item in enumerate(results):
        print(f"\nPrompt {i+1}: {test_data['prompt'][i]}")
        print(f"Response: {item['generations'][0]}")

if __name__ == "__main__":
    test_sglang_backend()