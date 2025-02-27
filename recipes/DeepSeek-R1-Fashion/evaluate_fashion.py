#!/usr/bin/env python
"""
Evaluation script for DeepSeek-R1-Fashion model.

This script evaluates the performance of the DeepSeek-R1-Fashion model on
various fashion-related tasks, including style recommendations, outfit compatibility,
trend analysis, and fashion knowledge.
"""

import os
import argparse
import json
from tqdm import tqdm
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams

@dataclass
class FashionEvalResult:
    """Result of evaluating a fashion model response."""
    task: str
    query: str
    response: str
    scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
@dataclass
class FashionEvalMetrics:
    """Aggregated metrics for fashion evaluation."""
    task: str
    avg_score: float
    score_breakdown: Dict[str, float]
    sample_count: int

class FashionEvaluator:
    """Evaluate fashion model performance."""
    
    def __init__(self, model_name_or_path, use_vllm=True, device=None):
        self.model_name = model_name_or_path
        
        # Setup evaluation data
        self.eval_tasks = {
            "style_advice": self._get_style_advice_queries(),
            "outfit_compatibility": self._get_outfit_compatibility_queries(),
            "trend_analysis": self._get_trend_analysis_queries(),
            "fashion_knowledge": self._get_fashion_knowledge_queries(),
        }
        
        # Setup model
        if use_vllm:
            self.vllm = True
            self.model = LLM(model=model_name_or_path)
            self.tokenizer = None  # Not needed with vLLM
            self.sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.9,
                max_tokens=1024,
            )
        else:
            self.vllm = False
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, 
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map=self.device
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device
            )
    
    def _format_prompt(self, query):
        """Format a prompt for the model."""
        system_prompt = """You are a helpful AI Assistant specialized in fashion advice. 
        You provide thoughtful fashion recommendations based on user requests.
        """
        
        return f"{system_prompt}\n\nUser: {query}\n\nAssistant:"
    
    def _generate_response(self, prompt):
        """Generate a response from the model."""
        if self.vllm:
            outputs = self.model.generate([prompt], self.sampling_params)
            return outputs[0].outputs[0].text
        else:
            outputs = self.pipe(
                prompt,
                max_new_tokens=1024,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )
            return outputs[0]["generated_text"][len(prompt):]
    
    def _analyze_response(self, task, query, response):
        """Analyze a model response for a given task."""
        # Initialize scores dictionary
        scores = {}
        
        # Basic relevance check
        fashion_terms = [
            "outfit", "style", "trend", "fashion", "clothes", "clothing",
            "accessory", "accessories", "color", "pattern", "wear"
        ]
        relevance_score = sum(1 for term in fashion_terms if term.lower() in response.lower()) / len(fashion_terms)
        scores["relevance"] = min(1.0, relevance_score * 2)  # Scale up to max of 1.0
        
        # Task-specific scoring
        if task == "style_advice":
            # Check for personalization and specific recommendations
            personalization = any(term in response.lower() for term in 
                                 ["your body", "your style", "your occasion", "your preference"])
            specific_items = sum(1 for term in 
                                ["shirt", "pants", "dress", "skirt", "jeans", "jacket", "coat", "shoes", "blouse", "suit"] 
                                if term.lower() in response.lower())
            
            scores["personalization"] = 1.0 if personalization else 0.0
            scores["specificity"] = min(1.0, specific_items / 5)
            
        elif task == "outfit_compatibility":
            # Check for explanation of why items work together
            explanation_terms = ["complement", "match", "pair", "work with", "goes with", "coordinate"]
            has_explanation = any(term in response.lower() for term in explanation_terms)
            
            color_discussion = any(color in response.lower() for color in 
                                  ["color", "tone", "shade", "hue", "contrast", "complement"])
            
            scores["explanation"] = 1.0 if has_explanation else 0.0
            scores["color_awareness"] = 1.0 if color_discussion else 0.0
            
        elif task == "trend_analysis":
            # Check for temporal awareness and specific trends
            temporal_terms = ["current", "season", "this year", "recent", "upcoming", "latest"]
            has_temporal = any(term in response.lower() for term in temporal_terms)
            
            trend_count = sum(1 for term in 
                             ["trending", "popular", "runway", "designer", "collection", "fashion week"] 
                             if term.lower() in response.lower())
            
            scores["temporal_awareness"] = 1.0 if has_temporal else 0.0
            scores["trend_specificity"] = min(1.0, trend_count / 3)
            
        elif task == "fashion_knowledge":
            # Check for historical references and technical terms
            historical = any(term in response.lower() for term in 
                           ["history", "traditional", "classic", "origin", "decade", "century", "era"])
            
            technical_terms = sum(1 for term in 
                                ["silhouette", "cut", "drape", "textile", "fabric", "stitch", "tailoring"] 
                                if term.lower() in response.lower())
            
            scores["historical_context"] = 1.0 if historical else 0.0
            scores["technical_knowledge"] = min(1.0, technical_terms / 3)
        
        # Calculate average score
        avg_score = sum(scores.values()) / len(scores)
        scores["average"] = avg_score
        
        return scores
    
    def evaluate(self, task=None, output_path=None):
        """
        Evaluate the model on fashion tasks.
        
        Args:
            task: Specific task to evaluate, or None for all tasks
            output_path: Path to save evaluation results
            
        Returns:
            Dictionary of evaluation results
        """
        tasks = [task] if task else list(self.eval_tasks.keys())
        all_results = []
        
        for task in tasks:
            print(f"Evaluating task: {task}")
            queries = self.eval_tasks[task]
            
            for query in tqdm(queries, desc=f"Evaluating {task}"):
                prompt = self._format_prompt(query)
                response = self._generate_response(prompt)
                scores = self._analyze_response(task, query, response)
                
                result = FashionEvalResult(
                    task=task,
                    query=query,
                    response=response,
                    scores=scores
                )
                all_results.append(result)
        
        # Compute aggregated metrics
        metrics = self._compute_metrics(all_results)
        
        # Save results if output path is provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                results_dict = {
                    "results": [asdict(r) for r in all_results],
                    "metrics": [asdict(m) for m in metrics]
                }
                json.dump(results_dict, f, indent=2)
        
        return metrics, all_results
    
    def _compute_metrics(self, results):
        """Compute aggregated metrics from evaluation results."""
        task_results = defaultdict(list)
        for result in results:
            task_results[result.task].append(result)
        
        metrics = []
        for task, task_results_list in task_results.items():
            # Collect all scores for this task
            all_scores = defaultdict(list)
            for result in task_results_list:
                for score_name, score_value in result.scores.items():
                    all_scores[score_name].append(score_value)
            
            # Compute average scores
            avg_scores = {
                score_name: sum(scores) / len(scores) 
                for score_name, scores in all_scores.items()
            }
            
            metrics.append(FashionEvalMetrics(
                task=task,
                avg_score=avg_scores["average"],
                score_breakdown=avg_scores,
                sample_count=len(task_results_list)
            ))
        
        return metrics
    
    def _get_style_advice_queries(self):
        """Get queries for style advice task."""
        return [
            "What should I wear to a summer wedding?",
            "How can I dress professionally while staying comfortable?",
            "I have a pear-shaped body. What styles would flatter my figure?",
            "What's a good casual outfit for a first date?",
            "How should I dress for a job interview in a tech company?",
            "What are some stylish outfits for rainy weather?",
            "How can I make a basic t-shirt and jeans look more fashionable?",
            "What should I pack for a weekend beach trip?",
            "How can I transition my summer wardrobe to fall?",
            "What are good outfit options for a plus-size figure?"
        ]
    
    def _get_outfit_compatibility_queries(self):
        """Get queries for outfit compatibility task."""
        return [
            "Do black pants go with a navy blue top?",
            "What colors complement a burgundy dress?",
            "What type of shoes would work with wide-leg pants?",
            "How can I mix patterns in an outfit without clashing?",
            "What accessories would enhance a simple white dress?",
            "Can I wear gold and silver jewelry together?",
            "What type of jacket would work with a floral midi skirt?",
            "How do I coordinate colors in a three-piece outfit?",
            "What bottom would pair well with an oversized sweater?",
            "How can I style a statement piece without overwhelming my outfit?"
        ]
    
    def _get_trend_analysis_queries(self):
        """Get queries for trend analysis task."""
        return [
            "What are the biggest fashion trends this season?",
            "Are skinny jeans still in style?",
            "What color palettes are trending for summer?",
            "How are sustainable fashion trends evolving?",
            "What accessories are popular right now?",
            "What vintage styles are making a comeback?",
            "How are gender-fluid fashion trends developing?",
            "What are the emerging streetwear trends?",
            "How are workplace fashion trends changing post-pandemic?",
            "What's the forecast for next season's fashion trends?"
        ]
    
    def _get_fashion_knowledge_queries(self):
        """Get queries for fashion knowledge task."""
        return [
            "What's the difference between haute couture and ready-to-wear?",
            "Can you explain what a capsule wardrobe is?",
            "What are the basic types of dress silhouettes?",
            "What's the history of the little black dress?",
            "How do different fabrics affect the drape of clothing?",
            "What are the rules of color theory in fashion?",
            "What's the difference between fashion and style?",
            "How has men's formal wear evolved over the last century?",
            "What makes a garment considered 'luxury'?",
            "How do seasonal color analyses work?"
        ]


def main():
    parser = argparse.ArgumentParser(description="Evaluate fashion model")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--output", type=str, default="fashion_eval_results.json", help="Output file path")
    parser.add_argument("--task", type=str, default=None, help="Specific task to evaluate")
    parser.add_argument("--no-vllm", action="store_true", help="Disable vLLM for generation")
    args = parser.parse_args()
    
    evaluator = FashionEvaluator(args.model, use_vllm=not args.no_vllm)
    metrics, results = evaluator.evaluate(task=args.task, output_path=args.output)
    
    # Print summary
    print("\n=== Fashion Evaluation Results ===")
    for metric in metrics:
        print(f"\nTask: {metric.task}")
        print(f"Average Score: {metric.avg_score:.2f}")
        print("Score Breakdown:")
        for name, score in metric.score_breakdown.items():
            if name != "average":
                print(f"  - {name}: {score:.2f}")
    
    print(f"\nDetailed results saved to {args.output}")


if __name__ == "__main__":
    main()
