"""
Persona Prediction System - Main Training and Demo Script
Generates synthetic data, trains multi-output model, and demonstrates inference.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_generator import generate_dataset, save_dataset
from src.feature_pipeline import FeaturePipeline
from src.model import create_model, ModelTrainer


def main():
    """Main entry point for training and demo"""
    
    print("=" * 60)
    print("PERSONA PREDICTION SYSTEM - DEMO")
    print("=" * 60)
    
    # Create directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # ============================================================
    # PHASE 1: GENERATE SYNTHETIC DATASET
    # ============================================================
    print("\n" + "=" * 60)
    print("PHASE 1: SYNTHETIC DATA GENERATION")
    print("=" * 60)
    
    data_path = "data/synthetic_dataset.csv"
    
    if os.path.exists(data_path):
        print(f"Loading existing dataset from {data_path}")
        df = pd.read_csv(data_path)
    else:
        print("Generating 500 synthetic samples...")
        df = generate_dataset(500)
        save_dataset(df, data_path)
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Show sample
    print("\n--- Sample Data (first 3 rows) ---")
    print(df.head(3).to_string())
    
    # Show distributions
    print("\n--- Intent Distribution ---")
    print(df["intent"].value_counts())
    
    print("\n--- Agent Emotion Distribution ---")
    print(df["agent_emotion"].value_counts())
    
    # ============================================================
    # PHASE 2: TRAIN MODEL
    # ============================================================
    print("\n" + "=" * 60)
    print("PHASE 2: MODEL TRAINING")
    print("=" * 60)
    
    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"\nTrain size: {len(train_df)}, Test size: {len(test_df)}")
    
    # Fit pipeline on training data
    print("\nFitting feature pipeline...")
    pipeline = FeaturePipeline(max_tfidf_features=100)
    pipeline.fit(train_df)
    
    # Transform features
    X_train = pipeline.transform_features(train_df)
    X_test = pipeline.transform_features(test_df)
    y_train = pipeline.transform_labels(train_df)
    y_test = pipeline.transform_labels(test_df)
    
    print(f"Input dimension: {pipeline.get_input_dim()}")
    print(f"Train features shape: {X_train.shape}")
    
    # Create model
    input_dim = pipeline.get_input_dim()
    output_dims = pipeline.get_output_dims()
    
    print("\n--- Output Dimensions ---")
    for name, dim in output_dims.items():
        print(f"  {name}: {dim} classes")
    
    model = create_model(
        input_dim=input_dim,
        output_dims=output_dims,
        hidden_dims=[256, 128, 64],  # Deeper network
        dropout=0.2  # Less dropout
    )
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")
    
    # Train
    print("\n--- Training ---")
    trainer = ModelTrainer(model, learning_rate=0.001)
    losses = trainer.fit(X_train, y_train, epochs=100, batch_size=32, verbose=True)
    
    # ============================================================
    # EVALUATION
    # ============================================================
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    
    # Convert test labels to tensors
    y_test_tensors = {name: torch.LongTensor(labels) for name, labels in y_test.items()}
    X_test_tensor = torch.FloatTensor(X_test)
    
    # Evaluate
    accuracies = trainer.evaluate(X_test_tensor, y_test_tensors)
    
    print("\n--- Per-Field Accuracy ---")
    for name, acc in accuracies.items():
        print(f"  {name}: {acc:.2%}")
    
    avg_accuracy = np.mean(list(accuracies.values()))
    print(f"\n  AVERAGE: {avg_accuracy:.2%}")
    
    # Save model and pipeline
    print("\n--- Saving Model ---")
    pipeline.save("models/pipeline.pkl")
    trainer.save("models/model.pt")
    
    # ============================================================
    # DEMO INFERENCE
    # ============================================================
    print("\n" + "=" * 60)
    print("DEMO INFERENCE")
    print("=" * 60)
    
    # Load predictor
    from src.inference import PersonaPredictor
    
    predictor = PersonaPredictor(
        pipeline_path="models/pipeline.pkl",
        model_path="models/model.pt"
    )
    
    # Test messages
    test_messages = [
        {
            "message": "This is absolutely unacceptable! I've been waiting for 5 days!",
            "expected": "Angry user → empathetic/sad agent emotion, calm tone, high empathy"
        },
        {
            "message": "Could you please explain how the billing works?",
            "expected": "Neutral inquiry → neutral agent emotion, neutral tone"
        },
        {
            "message": "I'm so worried this won't be fixed in time. Please help!",
            "expected": "Anxious user → calm/reassuring agent emotion, high reassurance"
        },
        {
            "message": "Thank you so much! Your team was amazing!",
            "expected": "Happy user → friendly agent emotion, friendly tone"
        },
        {
            "message": "I want to speak to a supervisor immediately!",
            "expected": "Escalation → high formality, apology required"
        },
    ]
    
    for test in test_messages:
        print(f"\n--- Input Message ---")
        print(f'"{test["message"]}"')
        print(f"Expected: {test['expected']}")
        
        result = predictor.predict_from_message_only(test["message"])
        
        print("\n--- Predicted Agent Behavior ---")
        for key, value in result.items():
            print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print("\nFiles created:")
    print("  - data/synthetic_dataset.csv (500 samples)")
    print("  - models/pipeline.pkl (feature pipeline)")
    print("  - models/model.pt (trained model)")
    print("\nTo use for inference:")
    print("  from src.inference import predict_persona")
    print('  result = predict_persona("Your message here")')


if __name__ == "__main__":
    main()
