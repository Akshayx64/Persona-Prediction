"""
Feature Processing Pipeline for Persona Prediction System
Handles preprocessing of numeric, categorical, and text features.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict, Tuple, List
import pickle
import os


class FeaturePipeline:
    """
    Preprocesses input features for the multi-output model.
    Handles:
    - Numeric features: StandardScaler
    - Categorical features: LabelEncoder + one-hot
    - Text features: TF-IDF vectorization
    """
    
    def __init__(self, max_tfidf_features: int = 100):
        """
        Args:
            max_tfidf_features: Maximum number of TF-IDF features to extract
        """
        self.max_tfidf_features = max_tfidf_features
        
        # Numeric features
        self.numeric_cols = [
            "anger_level", "frustration_level", "anxiety_level", "calm_level",
            "sentiment_negative", "sentiment_neutral", "sentiment_positive"
        ]
        self.numeric_scaler = StandardScaler()
        
        # Categorical features
        self.categorical_cols = ["intent", "urgency"]
        self.label_encoders = {}
        self.categorical_classes = {}
        
        # Text features
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_tfidf_features,
            stop_words="english",
            ngram_range=(1, 2)
        )
        
        # Output label encoders
        self.output_cols = [
            "agent_emotion", "tone", "empathy_level", "verbosity", "formality",
            "assertiveness", "apology_required", "reassurance_level", 
            "solution_speed", "boundary_setting", "explanation_depth"
        ]
        self.output_encoders = {}
        self.output_classes = {}
        
        self.is_fitted = False
    
    def fit(self, df: pd.DataFrame) -> "FeaturePipeline":
        """
        Fit the pipeline on training data.
        
        Args:
            df: DataFrame with all input features and output labels
        """
        # Fit numeric scaler
        self.numeric_scaler.fit(df[self.numeric_cols])
        
        # Fit categorical encoders
        for col in self.categorical_cols:
            le = LabelEncoder()
            le.fit(df[col])
            self.label_encoders[col] = le
            self.categorical_classes[col] = list(le.classes_)
        
        # Fit TF-IDF on text
        self.tfidf_vectorizer.fit(df["user_message"])
        
        # Fit output label encoders
        for col in self.output_cols:
            le = LabelEncoder()
            # Convert boolean to string for consistent encoding
            values = df[col].astype(str) if df[col].dtype == bool else df[col]
            le.fit(values)
            self.output_encoders[col] = le
            self.output_classes[col] = list(le.classes_)
        
        self.is_fitted = True
        return self
    
    def transform_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform input features into model-ready format.
        
        Args:
            df: DataFrame with input features
            
        Returns:
            Concatenated feature array
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before transform")
        
        # Transform numeric features
        numeric_features = self.numeric_scaler.transform(df[self.numeric_cols])
        
        # Transform categorical features (one-hot encoded)
        categorical_features = []
        for col in self.categorical_cols:
            encoded = self.label_encoders[col].transform(df[col])
            # One-hot encode
            n_classes = len(self.categorical_classes[col])
            one_hot = np.zeros((len(df), n_classes))
            one_hot[np.arange(len(df)), encoded] = 1
            categorical_features.append(one_hot)
        categorical_features = np.hstack(categorical_features)
        
        # Transform text features
        text_features = self.tfidf_vectorizer.transform(df["user_message"]).toarray()
        
        # Concatenate all features
        all_features = np.hstack([numeric_features, categorical_features, text_features])
        
        return all_features.astype(np.float32)
    
    def transform_labels(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Transform output labels into model-ready format.
        
        Args:
            df: DataFrame with output labels
            
        Returns:
            Dictionary mapping output column names to encoded labels
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before transform")
        
        labels = {}
        for col in self.output_cols:
            values = df[col].astype(str) if df[col].dtype == bool else df[col]
            labels[col] = self.output_encoders[col].transform(values)
        
        return labels
    
    def transform_single_message(self, message: str, features: Dict) -> np.ndarray:
        """
        Transform a single message + features for inference.
        
        Args:
            message: User message text
            features: Dict with numeric and categorical features
            
        Returns:
            Feature array ready for model input
        """
        # Create single-row DataFrame
        data = {"user_message": message, **features}
        df = pd.DataFrame([data])
        
        return self.transform_features(df)
    
    def decode_predictions(self, predictions: Dict[str, int]) -> Dict[str, str]:
        """
        Decode model predictions back to original labels.
        
        Args:
            predictions: Dict mapping output names to predicted class indices
            
        Returns:
            Dict mapping output names to decoded string labels
        """
        decoded = {}
        for col, pred in predictions.items():
            decoded[col] = self.output_encoders[col].inverse_transform([pred])[0]
            # Convert 'True'/'False' strings back to boolean
            if col in ["apology_required", "boundary_setting"]:
                decoded[col] = decoded[col] == "True"
        return decoded
    
    def get_input_dim(self) -> int:
        """Get the total input dimension after feature transformation"""
        numeric_dim = len(self.numeric_cols)  # 7
        categorical_dim = sum(len(classes) for classes in self.categorical_classes.values())  # 5 + 3 = 8
        text_dim = self.max_tfidf_features  # 100
        return numeric_dim + categorical_dim + text_dim
    
    def get_output_dims(self) -> Dict[str, int]:
        """Get the number of classes for each output"""
        return {col: len(classes) for col, classes in self.output_classes.items()}
    
    def save(self, filepath: str) -> None:
        """Save pipeline to file"""
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        print(f"Pipeline saved to {filepath}")
    
    @staticmethod
    def load(filepath: str) -> "FeaturePipeline":
        """Load pipeline from file"""
        with open(filepath, "rb") as f:
            return pickle.load(f)


if __name__ == "__main__":
    # Test the pipeline
    from data_generator import generate_dataset
    
    print("Testing Feature Pipeline...")
    
    # Generate small test dataset
    df = generate_dataset(100)
    
    # Fit pipeline
    pipeline = FeaturePipeline(max_tfidf_features=50)
    pipeline.fit(df)
    
    # Transform features
    X = pipeline.transform_features(df)
    y = pipeline.transform_labels(df)
    
    print(f"\nInput dimension: {pipeline.get_input_dim()}")
    print(f"Feature array shape: {X.shape}")
    print(f"\nOutput dimensions:")
    for col, dim in pipeline.get_output_dims().items():
        print(f"  {col}: {dim} classes")
