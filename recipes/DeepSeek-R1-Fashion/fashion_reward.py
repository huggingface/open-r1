"""
Fashion Relevance Reward Function for DeepSeek-R1-Fashion model.

This module implements a custom reward function that evaluates the fashion relevance
and quality of responses for the GRPO training phase.
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from open_r1.reward.base import RewardFunction

class FashionRelevanceReward(RewardFunction):
    """
    Reward function that evaluates fashion-specific qualities in responses:
    1. Fashion terminology usage
    2. Personalization
    3. Practicality of advice
    4. Multi-option recommendations
    5. Style explanation
    """
    
    def __init__(self):
        super().__init__()
        # Fashion-related terminology to look for
        self.fashion_terms = [
            "outfit", "style", "trend", "accessory", "accessories", "color", "pattern",
            "fabric", "silhouette", "wardrobe", "dressy", "casual", "formal", "fit",
            "texture", "layering", "seasonal", "classic", "contemporary", "vintage",
            "sustainable", "tailored", "oversized", "minimalist", "statement", "aesthetic"
        ]
        
    def _calculate_fashion_term_score(self, text: str) -> float:
        """Calculate score based on fashion terminology usage."""
        text = text.lower()
        term_count = sum(1 for term in self.fashion_terms if term in text)
        # Normalize the score to [0, 1] range with diminishing returns
        return min(1.0, term_count / 10)
        
    def _calculate_personalization_score(self, text: str) -> float:
        """Calculate score based on personalization indicators."""
        personalization_patterns = [
            r"body type", r"skin tone", r"personal style", r"preference",
            r"comfort", r"occasion", r"your", r"you might", r"you could",
            r"depending on", r"for your", r"based on your"
        ]
        
        count = sum(1 for pattern in personalization_patterns if re.search(pattern, text, re.IGNORECASE))
        return min(1.0, count / 5)
        
    def _calculate_practicality_score(self, text: str) -> float:
        """Calculate score based on practical advice indicators."""
        practicality_patterns = [
            r"budget", r"affordable", r"investment", r"versatile", r"mix and match",
            r"capsule", r"staple", r"essential", r"weather", r"season", r"occasion",
            r"day-to-night", r"transition", r"maintenance", r"wash", r"care"
        ]
        
        count = sum(1 for pattern in practicality_patterns if re.search(pattern, text, re.IGNORECASE))
        return min(1.0, count / 5)
        
    def _calculate_options_score(self, text: str) -> float:
        """Calculate score based on providing multiple options."""
        # Look for numbered lists, bullet points, or option indicators
        option_patterns = [
            r"\d+\.", r"\*", r"-", r"option", r"alternative", r"another",
            r"additionally", r"other", r"second", r"third", r"first", r"instead"
        ]
        
        count = sum(1 for pattern in option_patterns if re.search(pattern, text, re.IGNORECASE))
        return min(1.0, count / 5)
        
    def _calculate_style_explanation_score(self, text: str) -> float:
        """Calculate score based on style explanation."""
        explanation_patterns = [
            r"because", r"reason", r"complement", r"enhance", r"flattering",
            r"highlight", r"balance", r"proportion", r"elongate", r"slimming",
            r"emphasize", r"contrast", r"coordinate", r"this works", r"this helps"
        ]
        
        count = sum(1 for pattern in explanation_patterns if re.search(pattern, text, re.IGNORECASE))
        return min(1.0, count / 5)
    
    def compute_reward(
        self,
        completion: str,
        prompt: Optional[str] = None,
        prompt_metadata: Optional[Dict[str, Any]] = None,
        completion_metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute the fashion relevance reward score.
        
        Args:
            completion: The model's generated completion
            prompt: The input prompt
            prompt_metadata: Additional metadata about the prompt
            completion_metadata: Additional metadata about the completion
            
        Returns:
            A tuple of (reward_score, metadata_dict)
        """
        # Initialize metadata dictionary
        metadata = {
            "fashion_term_score": 0.0,
            "personalization_score": 0.0,
            "practicality_score": 0.0,
            "options_score": 0.0,
            "style_explanation_score": 0.0,
        }
        
        # Skip if completion is empty
        if not completion or len(completion.strip()) == 0:
            return 0.0, metadata
        
        # Calculate component scores
        metadata["fashion_term_score"] = self._calculate_fashion_term_score(completion)
        metadata["personalization_score"] = self._calculate_personalization_score(completion)
        metadata["practicality_score"] = self._calculate_practicality_score(completion)
        metadata["options_score"] = self._calculate_options_score(completion)
        metadata["style_explanation_score"] = self._calculate_style_explanation_score(completion)
        
        # Compute overall score (weighted average)
        weights = {
            "fashion_term_score": 1.0,
            "personalization_score": 1.5,
            "practicality_score": 1.0,
            "options_score": 0.8,
            "style_explanation_score": 1.2,
        }
        
        total_weight = sum(weights.values())
        overall_score = sum(metadata[k] * weights[k] for k in metadata) / total_weight
        
        # Add overall score to metadata
        metadata["overall_score"] = overall_score
        
        return overall_score, metadata
