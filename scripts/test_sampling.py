from datasets import Dataset
from open_r1.generate import build_distilabel_pipeline

# To run this test:
# First ensure both backends are running:
# For vLLM:
#   python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-2-7b-chat-hf --port 8000
#
# For SGLang (in a different terminal):
#   sglang start-server --model meta-llama/Llama-2-7b-chat-hf --port 8001
#
# Then run: python test_sampling.py
#
# This test script:
# - Tests four different sampling configurations:
#   - High temperature (more random)
#   - Low temperature (more deterministic)
#   - Top-p sampling
#   - Combined temperature and top-p
# - Generates multiple responses for each configuration to observe variation
# - Uses a single prompt that should produce creative, varied responses to make sampling differences more apparent


def test_sampling_behavior(backend: str):
    # Single prompt to test sampling variations
    test_data = {
        "prompt": [
            "Write a creative color name and briefly describe it.",  # This prompt should generate varied responses with sampling
        ]
    }
    test_dataset = Dataset.from_dict(test_data)

    # Test different sampling configurations
    sampling_configs = [
        {
            "name": "High Temperature",
            "params": {"temperature": 0.9, "top_p": None}
        },
        {
            "name": "Low Temperature",
            "params": {"temperature": 0.1, "top_p": None}
        },
        {
            "name": "Top-P Sampling",
            "params": {"temperature": None, "top_p": 0.9}
        },
        {
            "name": "Combined Sampling",
            "params": {"temperature": 0.7, "top_p": 0.9}
        },
    ]

    print(f"\nTesting {backend} backend sampling behavior:")
    print("=" * 50)

    for config in sampling_configs:
        print(f"\nTesting {config['name']}:")
        print("-" * 30)

        # Create pipeline with specific sampling parameters
        pipeline = build_distilabel_pipeline(
            model="meta-llama/Llama-2-7b-chat-hf",  # Replace with your model
            backend=backend,
            base_url="http://localhost:8000",
            prompt_column="prompt",
            temperature=config['params']['temperature'],
            top_p=config['params']['top_p'],
            max_new_tokens=50,  # Keep short for testing
            num_generations=3,  # Generate multiple responses to see variation
            input_batch_size=1,
            timeout=30,
        )

        # Run inference
        results = pipeline.run(
            dataset=test_dataset,
            dataset_batch_size=1,
            use_cache=False,
        )

        # Print all generations for this configuration
        for item in results:
            print("\nGenerations:")
            for i, gen in enumerate(item['generations'], 1):
                print(f"  {i}. {gen}")


def main():
    # Test both backends
    for backend in ["vllm", "sglang"]:
        try:
            test_sampling_behavior(backend)
        except Exception as e:
            print(f"\nError testing {backend} backend: {str(e)}")


if __name__ == "__main__":
    main()