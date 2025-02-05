import pytest
import uuid
from datasets import Dataset
from open_r1.generate import build_distilabel_pipeline

@pytest.fixture
def test_dataset():
    """Create a small test dataset."""
    return Dataset.from_dict({
        "prompt": [
            "What is 2+2?",
            "Explain quantum computing in one sentence.",
            "Write a haiku about programming."
        ]
    })

@pytest.mark.parametrize("backend", ["vllm", "sglang"])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_basic_inference(test_dataset, backend, batch_size):
    """Test basic inference capabilities with different batch sizes."""
    unique_id = str(uuid.uuid4())[:8]
    
    generation_kwargs = {
        "temperature": 0.0,  # Deterministic for testing
        "max_new_tokens": 50,
        "top_p": 1.0
    }
    
    pipeline = build_distilabel_pipeline(
        model="meta-llama/Llama-2-7b-chat-hf",
        backend=backend,
        prompt_column="prompt",
        generation_kwargs=generation_kwargs,
        input_batch_size=batch_size,
        unique_id=unique_id,
        num_generations=1
    )
    
    result = pipeline.run(
        dataset=test_dataset,
        dataset_batch_size=batch_size,
        use_cache=False,
    )
    
    # Basic validation
    assert len(result) == len(test_dataset)
    assert "text" in result.features
    assert all(isinstance(gen, str) and len(gen) > 0 for gen in result["text"])

@pytest.mark.parametrize("backend", ["vllm", "sglang"])
def test_error_handling(test_dataset, backend):
    """Test error handling with invalid configurations."""
    with pytest.raises(ValueError):
        # Test with invalid batch size
        pipeline = build_distilabel_pipeline(
            model="meta-llama/Llama-2-7b-chat-hf",
            backend=backend,
            input_batch_size=0
        )

@pytest.mark.parametrize("backend", ["vllm", "sglang"])
def test_multiple_generations(test_dataset, backend):
    """Test multiple generations per prompt."""
    unique_id = str(uuid.uuid4())[:8]
    
    generation_kwargs = {
        "temperature": 0.7,  # Non-deterministic for multiple generations
        "max_new_tokens": 50,
        "top_p": 0.95
    }
    
    pipeline = build_distilabel_pipeline(
        model="meta-llama/Llama-2-7b-chat-hf",
        backend=backend,
        prompt_column="prompt",
        generation_kwargs=generation_kwargs,
        input_batch_size=1,
        unique_id=unique_id,
        num_generations=2  # Generate 2 responses per prompt
    )
    
    result = pipeline.run(
        dataset=test_dataset,
        dataset_batch_size=1,
        use_cache=False,
    )
    
    # Verify multiple generations
    assert len(result) == len(test_dataset) * 2
    assert "text" in result.features

@pytest.mark.parametrize("backend", ["vllm", "sglang"])
def test_generation_parameters(test_dataset, backend):
    """Test different generation parameters."""
    generation_configs = [
        {"temperature": 0.0, "max_new_tokens": 10, "top_p": 1.0},
        {"temperature": 0.8, "max_new_tokens": 100, "top_p": 0.9},
    ]
    
    for gen_kwargs in generation_configs:
        pipeline = build_distilabel_pipeline(
            model="meta-llama/Llama-2-7b-chat-hf",
            backend=backend,
            prompt_column="prompt",
            generation_kwargs=gen_kwargs,
            input_batch_size=1
        )
        
        result = pipeline.run(
            dataset=test_dataset,
            dataset_batch_size=1,
            use_cache=False,
        )
        
        assert len(result) == len(test_dataset)
        if gen_kwargs["temperature"] == 0.0:
            # For deterministic generation, outputs should be identical across runs
            second_result = pipeline.run(
                dataset=test_dataset,
                dataset_batch_size=1,
                use_cache=False,
            )
            assert all(r1 == r2 for r1, r2 in zip(result["text"], second_result["text"]))