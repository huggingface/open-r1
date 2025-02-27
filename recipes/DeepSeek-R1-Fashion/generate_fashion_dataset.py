#!/usr/bin/env python
"""
Script to generate a fashion dataset for training DeepSeek-R1-Fashion model.

This script uses Distilabel to generate fashion-related conversations and recommendations
that can be used for fine-tuning the DeepSeek-R1 model for fashion tasks.
"""

import os
import argparse
from datasets import Dataset
from distilabel.pipeline import Pipeline
from distilabel.tasks import ChatGeneration
from distilabel.steps.llm import HuggingFaceLLM, VLLMStep
from distilabel.steps.prompt import PromptTemplate

def parse_args():
    parser = argparse.ArgumentParser(description="Generate fashion dataset for DeepSeek-R1")
    parser.add_argument("--output-path", type=str, default="data/fashion-dataset",
                       help="Path to save the generated dataset")
    parser.add_argument("--num-samples", type=int, default=1000,
                       help="Number of samples to generate")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1",
                       help="Model to use for generation")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Fashion-related queries
    fashion_queries = [
        "What's a good outfit for a summer wedding?",
        "How do I style a basic white t-shirt?",
        "What are the key fashion trends for Fall 2025?",
        "Can you recommend sustainable fashion brands?",
        "How should I dress for a job interview in tech?",
        "What accessories go well with a little black dress?",
        "How do I build a minimalist wardrobe?",
        "What colors are complementary to olive skin tone?",
        "How do I style oversized clothing without looking sloppy?",
        "What's the difference between business casual and smart casual?",
        # Add more fashion queries here
    ]
    
    # Create dataset from queries
    query_dataset = Dataset.from_dict({"text": fashion_queries})
    
    # System prompt for fashion advice
    system_prompt = """You are a helpful AI assistant specializing in fashion advice.
    When responding to fashion-related queries, follow these guidelines:
    1. Consider the occasion, body type, personal style, and practical concerns
    2. Provide specific recommendations with reasoning
    3. Include options at different price points when appropriate
    4. Suggest styling combinations and accessories
    5. Mention current trends while respecting timeless principles
    
    Your advice should be detailed, personalized, and practical."""
    
    # Create generation pipeline
    template = PromptTemplate(
        template=system_prompt + "\n\nUser: {{text}}\nAssistant:",
        input_columns=["text"],
        output_column="response"
    )
    
    # Use VLLM for faster generation if available
    try:
        generator = VLLMStep(
            model=args.model,
            generation_kwargs={
                "max_new_tokens": 1024,
                "temperature": 0.7,
                "top_p": 0.9,
            }
        )
    except:
        # Fallback to HuggingFace generation
        generator = HuggingFaceLLM(
            model_id=args.model,
            generation_kwargs={
                "max_new_tokens": 1024,
                "temperature": 0.7,
                "top_p": 0.9,
            }
        )
    
    # Setup pipeline
    pipeline = Pipeline(
        steps={
            "template": template,
            "generator": generator,
        },
        connections={
            "template": ["generator"],
        },
        input_dataset=query_dataset,
        input_keys=["text"],
        output_keys=["response"],
    )
    
    # Run generation
    print(f"Generating {args.num_samples} fashion conversation samples...")
    results = pipeline.run(
        num_samples=args.num_samples,
        output_path=args.output_path,
        format="jsonl",
    )
    
    print(f"Dataset generation complete. Saved to {args.output_path}")
    print(f"Generated {len(results)} samples")
    
if __name__ == "__main__":
    main()
