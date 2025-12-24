"""
Inference Module for Persona Prediction System
Provides easy-to-use functions for predicting agent behavior from user messages.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional
import os

from .feature_pipeline import FeaturePipeline
from .model import MultiOutputModel, create_model


class PersonaPredictor:
    """
    High-level inference class for predicting agent behavior fields.
    Handles feature extraction, model inference, and result decoding.
    """
    
    def __init__(
        self,
        pipeline_path: str,
        model_path: str,
        device: str = None
    ):
        """
        Args:
            pipeline_path: Path to saved FeaturePipeline
            model_path: Path to saved model checkpoint
            device: Device to use for inference
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load pipeline
        self.pipeline = FeaturePipeline.load(pipeline_path)
        
        # Load checkpoint to get architecture config
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create model with saved architecture
        self.model = create_model(
            input_dim=checkpoint["input_dim"],
            output_dims=checkpoint["output_dims"],
            hidden_dims=checkpoint["hidden_dims"],
            dropout=checkpoint["dropout"]
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
    
    def predict(
        self,
        user_message: str,
        anger_level: float = 0.0,
        frustration_level: float = 0.0,
        anxiety_level: float = 0.0,
        calm_level: float = 0.5,
        sentiment_negative: float = 0.0,
        sentiment_neutral: float = 0.5,
        sentiment_positive: float = 0.5,
        intent: str = "inquiry",
        urgency: str = "medium"
    ) -> Dict[str, Any]:
        """
        Predict agent behavior fields for a given user message.
        
        Args:
            user_message: The user's support message
            anger_level: User anger level [0,1]
            frustration_level: User frustration level [0,1]
            anxiety_level: User anxiety level [0,1]
            calm_level: User calm level [0,1]
            sentiment_negative: Negative sentiment score [0,1]
            sentiment_neutral: Neutral sentiment score [0,1]
            sentiment_positive: Positive sentiment score [0,1]
            intent: User intent (complaint, inquiry, clarification, escalation_request, praise)
            urgency: Urgency level (low, medium, high)
            
        Returns:
            Dict with all predicted agent behavior fields
        """
        # Prepare features
        features = {
            "anger_level": anger_level,
            "frustration_level": frustration_level,
            "anxiety_level": anxiety_level,
            "calm_level": calm_level,
            "sentiment_negative": sentiment_negative,
            "sentiment_neutral": sentiment_neutral,
            "sentiment_positive": sentiment_positive,
            "intent": intent,
            "urgency": urgency,
        }
        
        # Transform features
        X = self.pipeline.transform_single_message(user_message, features)
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(X_tensor)
            predictions = {}
            for name, logits in outputs.items():
                predictions[name] = torch.argmax(logits, dim=1).item()
        
        # Decode predictions
        decoded = self.pipeline.decode_predictions(predictions)
        
        return decoded
    
    def predict_from_message_only(
        self,
        user_message: str,
        estimate_features: bool = True
    ) -> Dict[str, Any]:
        """
        Predict agent behavior from just the user message.
        Optionally estimates emotion/sentiment features from message content.
        
        Args:
            user_message: The user's support message
            estimate_features: If True, estimate features based on message keywords
            
        Returns:
            Dict with all predicted agent behavior fields
        """
        if estimate_features:
            features = self._estimate_features_from_message(user_message)
        else:
            # Use neutral defaults
            features = {
                "anger_level": 0.0,
                "frustration_level": 0.0,
                "anxiety_level": 0.0,
                "calm_level": 0.5,
                "sentiment_negative": 0.2,
                "sentiment_neutral": 0.5,
                "sentiment_positive": 0.3,
                "intent": "inquiry",
                "urgency": "medium",
            }
        
        return self.predict(user_message, **features)
    
    def _estimate_features_from_message(self, message: str) -> Dict:
        """
        Simple rule-based estimation of features from message content.
        Used when explicit feature values are not provided.
        """
        message_lower = message.lower()
        
        # Estimate anger
        anger_keywords = ["angry", "furious", "unacceptable", "ridiculous", "terrible", "worst", "hate"]
        anger = min(1.0, sum(0.2 for kw in anger_keywords if kw in message_lower))
        
        # Estimate frustration
        frustration_keywords = ["frustrated", "annoying", "waited", "still", "again", "times"]
        frustration = min(1.0, sum(0.15 for kw in frustration_keywords if kw in message_lower))
        
        # Estimate anxiety
        anxiety_keywords = ["worried", "concerned", "urgent", "help", "please", "need"]
        anxiety = min(1.0, sum(0.15 for kw in anxiety_keywords if kw in message_lower))
        
        # Estimate calm
        calm = max(0.0, 1.0 - anger - frustration)
        
        # Estimate sentiment
        negative_keywords = ["bad", "wrong", "problem", "issue", "broken", "fail"]
        positive_keywords = ["thank", "great", "excellent", "amazing", "perfect", "love"]
        
        neg_score = min(1.0, sum(0.15 for kw in negative_keywords if kw in message_lower))
        pos_score = min(1.0, sum(0.15 for kw in positive_keywords if kw in message_lower))
        neutral_score = max(0.0, 1.0 - neg_score - pos_score)
        
        # Estimate intent
        if any(kw in message_lower for kw in ["supervisor", "manager", "escalate"]):
            intent = "escalation_request"
        elif any(kw in message_lower for kw in ["thank", "great", "amazing", "excellent"]):
            intent = "praise"
        elif any(kw in message_lower for kw in ["explain", "understand", "confused", "clarify"]):
            intent = "clarification"
        elif any(kw in message_lower for kw in ["unacceptable", "refund", "complaint", "terrible"]):
            intent = "complaint"
        else:
            intent = "inquiry"
        
        # Estimate urgency
        if any(kw in message_lower for kw in ["immediately", "urgent", "asap", "now"]):
            urgency = "high"
        elif any(kw in message_lower for kw in ["soon", "quickly"]):
            urgency = "medium"
        else:
            urgency = "low"
        
        return {
            "anger_level": anger,
            "frustration_level": frustration,
            "anxiety_level": anxiety,
            "calm_level": calm,
            "sentiment_negative": neg_score,
            "sentiment_neutral": neutral_score,
            "sentiment_positive": pos_score,
            "intent": intent,
            "urgency": urgency,
        }


def load_predictor(
    model_dir: str = "models"
) -> PersonaPredictor:
    """
    Convenience function to load a predictor from default paths.
    
    Args:
        model_dir: Directory containing saved pipeline and model
        
    Returns:
        Configured PersonaPredictor instance
    """
    pipeline_path = os.path.join(model_dir, "pipeline.pkl")
    model_path = os.path.join(model_dir, "model.pt")
    
    return PersonaPredictor(pipeline_path, model_path)


# Standalone function for quick inference
def predict_persona(
    user_message: str,
    model_dir: str = "models"
) -> Dict[str, Any]:
    """
    Quick inference function - loads model and predicts.
    For production use, prefer creating a PersonaPredictor instance once.
    
    Args:
        user_message: The user's support message
        model_dir: Directory containing saved pipeline and model
        
    Returns:
        Dict with all predicted agent behavior fields
    """
    predictor = load_predictor(model_dir)
    return predictor.predict_from_message_only(user_message)


if __name__ == "__main__":
    print("Inference module loaded successfully.")
    print("Use PersonaPredictor class or predict_persona() function for inference.")
