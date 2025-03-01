#!/usr/bin/env python
"""
Simplified script to generate a fashion dataset for training DeepSeek-R1-Fashion model.
"""

import os
import json
import argparse
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset

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
        # Additional queries for variety
        "How can I dress professionally while pregnant?",
        "What are good outfit ideas for a first date?",
        "How do I choose the right jeans for my body type?",
        "What should I wear to a music festival?",
        "How do I transition my wardrobe from winter to spring?",
        "What are must-have pieces for a capsule wardrobe?",
        "How can I dress to look taller?",
        "What's appropriate to wear to a funeral?",
        "How do I care for silk clothing?",
        "What are some 90s fashion trends making a comeback?"
    ]
    
    # System prompt for fashion advice
    system_prompt = """You are a helpful AI assistant specializing in fashion advice.
    When responding to fashion-related queries, follow these guidelines:
    1. Consider the occasion, body type, personal style, and practical concerns
    2. Provide specific recommendations with reasoning
    3. Include options at different price points when appropriate
    4. Suggest styling combinations and accessories
    5. Mention current trends while respecting timeless principles
    
    Your advice should be detailed, personalized, and practical."""
    
    print("Loading tokenizer and model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Using a fallback model instead...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained("gpt2")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Generate responses
    print(f"Generating {args.num_samples} fashion conversation samples...")
    all_data = []
    for _ in tqdm(range(args.num_samples)):
        # Select a random query
        query = np.random.choice(fashion_queries)
        
        # Format the prompt
        prompt = f"{system_prompt}\n\nUser: {query}\nAssistant:"
        
        # Generate response
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the assistant's response
        try:
            assistant_response = response.split("Assistant:")[1].strip()
        except IndexError:
            assistant_response = response.replace(prompt, "").strip()
        
        # Store the data
        data = {
            "text": query,
            "response": assistant_response
        }
        all_data.append(data)
    
    # Save the dataset
    with open(args.output_path, 'w') as f:
        for item in all_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Dataset generation complete. Saved to {args.output_path}")
    print(f"Generated {len(all_data)} samples")
    
if __name__ == "__main__":
    main()
