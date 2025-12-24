# Persona Prediction System

A demo AI system that predicts **complementary agent behavior fields** for customer support interactions.

## Overview

This system analyzes customer support messages and predicts how a support agent should respond. The key insight is that agent emotions should **complement** (not mirror) user emotions:

| User State | Agent Response |
|------------|----------------|
| Angry | Empathetic, Sad |
| Frustrated | Empathetic |
| Anxious | Calm, Reassuring |
| Happy | Friendly |
| Neutral | Neutral |

## Features

- **500 synthetic training samples** with realistic support messages
- **11 agent behavior predictions**:
  - `agent_emotion`: empathetic, sad, calm, reassuring, friendly, neutral
  - `tone`: calm, friendly, neutral
  - `empathy_level`: low, medium, high
  - `verbosity`: low, medium, high
  - `formality`: low, high
  - `assertiveness`: low, medium, high
  - `apology_required`: true/false
  - `reassurance_level`: low, medium, high
  - `solution_speed`: slow, normal, fast
  - `boundary_setting`: true/false
  - `explanation_depth`: low, high

## Quick Start

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Training & Demo

```bash
python main.py
```

This will:
1. Generate 500 synthetic samples
2. Train the multi-output model
3. Evaluate accuracy per field
4. Run demo inference on test messages

### Use for Inference

```python
from src.inference import PersonaPredictor

predictor = PersonaPredictor(
    pipeline_path="models/pipeline.pkl",
    model_path="models/model.pt"
)

result = predictor.predict_from_message_only(
    "This is unacceptable! I've been waiting for days!"
)
print(result)
# {'agent_emotion': 'empathetic', 'tone': 'calm', 'empathy_level': 'high', ...}
```

## Project Structure

```
├── data/
│   └── synthetic_dataset.csv    # Generated training data
├── models/
│   ├── pipeline.pkl             # Feature preprocessing
│   └── model.pt                 # Trained model
├── src/
│   ├── data_generator.py        # Synthetic data generation
│   ├── feature_pipeline.py      # Feature preprocessing
│   ├── model.py                 # Multi-output neural network
│   └── inference.py             # Inference API
├── main.py                      # Training script
└── requirements.txt
```

## Model Architecture

- **Input**: 115 features (7 numeric + 8 one-hot encoded + 100 TF-IDF)
- **Hidden Layers**: [256, 128, 64] with BatchNorm, ReLU, Dropout
- **Output**: 11 separate heads (one per behavior field)
- **Parameters**: 73,824

## Label Generation Rules

The synthetic data follows these rules:
- High anger/frustration → calm tone + high empathy
- High urgency → low verbosity + fast solution
- High anxiety → high reassurance
- Complaint/escalation → high formality + apology required
- Clarification intent → high explanation depth
- Very high anger → boundary setting enabled


## Requirements

- Python 3.8+
- PyTorch
- scikit-learn
- pandas
- numpy
