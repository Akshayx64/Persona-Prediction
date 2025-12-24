"""
Synthetic Data Generator for Persona Prediction System
Generates 500 synthetic customer support messages with input features and agent behavior labels.
"""

import random
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)


# ============================================================
# MESSAGE TEMPLATES
# ============================================================

# Templates for different intents and emotional states
MESSAGE_TEMPLATES = {
    "complaint": [
        "This is unacceptable! I've been waiting for {days} days and still no response.",
        "I'm extremely disappointed with your service. My order #{order} is {issue}.",
        "Why hasn't anyone fixed my problem yet? This is the {nth} time I'm contacting you!",
        "Your product broke after just {days} days. I want a full refund immediately.",
        "I can't believe how poorly I've been treated. This is ridiculous!",
        "My account has been charged incorrectly {times} times now. Fix this!",
        "The quality of your service has gone downhill. Very frustrated customer here.",
        "I've spent {hours} hours trying to resolve this issue with no success.",
        "This is the worst customer experience I've ever had. Completely unacceptable.",
        "Nobody seems to care about my problem. I've been passed around {times} times.",
    ],
    "inquiry": [
        "Hi, could you please tell me the status of my order #{order}?",
        "I'd like to know more about your {product} service. Can you help?",
        "What are the available options for {topic}?",
        "I'm interested in learning about your pricing plans.",
        "Could you explain how the {feature} feature works?",
        "Is it possible to change my subscription to the {plan} plan?",
        "What's the expected delivery time for {location}?",
        "I have a question about my recent bill. Can someone explain it?",
        "How do I update my account information?",
        "What are your business hours for customer support?",
    ],
    "clarification": [
        "I don't understand the email you sent. Could you clarify what you mean?",
        "The instructions are confusing. Can you explain step {step} again?",
        "I followed your advice but it didn't work. What am I missing?",
        "Sorry, I'm not sure what you mean by '{term}'. Can you explain?",
        "The error message says '{error}'. What does this mean?",
        "I read the FAQ but I'm still confused about {topic}.",
        "Can you walk me through the process one more time?",
        "I'm not tech-savvy. Could you explain this in simpler terms?",
        "What exactly do I need to do to resolve this issue?",
        "I'm lost. Can you help me understand what went wrong?",
    ],
    "escalation_request": [
        "I need to speak to a supervisor immediately.",
        "This has gone on too long. Please escalate my case to management.",
        "I demand to speak with someone who can actually help me.",
        "Your representatives have been unhelpful. Get me a manager now.",
        "I want this escalated to your highest level. This is unacceptable.",
        "If this isn't resolved today, I'm taking this to consumer protection.",
        "Connect me to someone senior who has the authority to fix this.",
        "I've had enough of the runaround. Escalate this immediately.",
        "I want to file a formal complaint and speak to a supervisor.",
        "This needs management attention. Regular support isn't cutting it.",
    ],
    "praise": [
        "Thank you so much! {name} was incredibly helpful!",
        "I just wanted to say how impressed I am with your service.",
        "Your team resolved my issue so quickly. Amazing!",
        "I appreciate the prompt response. You guys are great!",
        "This was the best customer service experience I've ever had!",
        "Kudos to your team for going above and beyond!",
        "I'm so grateful for your help. You made my day!",
        "Excellent service! I'll definitely recommend you to others.",
        "I was worried, but you fixed everything perfectly. Thank you!",
        "Your support team is fantastic. Keep up the great work!",
    ],
}

# Placeholders for dynamic message generation
PLACEHOLDERS = {
    "days": ["2", "3", "5", "7", "10", "14"],
    "order": ["12345", "67890", "24680", "13579", "11111"],
    "issue": ["damaged", "missing", "delayed", "wrong item", "never arrived"],
    "nth": ["3rd", "4th", "5th", "6th"],
    "times": ["2", "3", "4", "5"],
    "hours": ["2", "3", "4", "5", "6"],
    "product": ["premium", "basic", "enterprise", "starter"],
    "topic": ["billing", "shipping", "returns", "upgrades"],
    "feature": ["auto-renewal", "two-factor authentication", "data export"],
    "plan": ["premium", "professional", "basic"],
    "location": ["international orders", "rural areas", "express delivery"],
    "step": ["2", "3", "4", "5"],
    "term": ["API key", "integration", "webhook", "SSO"],
    "error": ["Connection timeout", "Invalid credentials", "Access denied"],
    "name": ["Sarah", "John", "Maria", "David", "Emily"],
}


def fill_placeholders(template: str) -> str:
    """Fill template placeholders with random values"""
    result = template
    for key, values in PLACEHOLDERS.items():
        placeholder = "{" + key + "}"
        if placeholder in result:
            result = result.replace(placeholder, random.choice(values))
    return result


# ============================================================
# INPUT FEATURE GENERATION
# ============================================================

def generate_input_features() -> Dict:
    """
    Generate input features for a single datapoint.
    Returns dict with all input features.
    """
    # Random intent selection
    intent = random.choice(["complaint", "inquiry", "clarification", "escalation_request", "praise"])
    
    # Random urgency
    urgency = random.choice(["low", "medium", "high"])
    
    # Generate emotion levels based on intent (with some randomness)
    if intent == "complaint":
        anger = np.random.uniform(0.4, 1.0)
        frustration = np.random.uniform(0.4, 1.0)
        anxiety = np.random.uniform(0.1, 0.5)
        calm = np.random.uniform(0.0, 0.3)
    elif intent == "escalation_request":
        anger = np.random.uniform(0.5, 1.0)
        frustration = np.random.uniform(0.6, 1.0)
        anxiety = np.random.uniform(0.2, 0.6)
        calm = np.random.uniform(0.0, 0.2)
    elif intent == "inquiry":
        anger = np.random.uniform(0.0, 0.2)
        frustration = np.random.uniform(0.0, 0.2)
        anxiety = np.random.uniform(0.0, 0.4)
        calm = np.random.uniform(0.5, 1.0)
    elif intent == "clarification":
        anger = np.random.uniform(0.0, 0.3)
        frustration = np.random.uniform(0.1, 0.5)
        anxiety = np.random.uniform(0.2, 0.6)
        calm = np.random.uniform(0.3, 0.7)
    else:  # praise
        anger = np.random.uniform(0.0, 0.1)
        frustration = np.random.uniform(0.0, 0.1)
        anxiety = np.random.uniform(0.0, 0.2)
        calm = np.random.uniform(0.7, 1.0)
    
    # Generate sentiment scores based on intent
    if intent in ["complaint", "escalation_request"]:
        sentiment_negative = np.random.uniform(0.5, 1.0)
        sentiment_neutral = np.random.uniform(0.0, 0.3)
        sentiment_positive = np.random.uniform(0.0, 0.2)
    elif intent == "praise":
        sentiment_negative = np.random.uniform(0.0, 0.1)
        sentiment_neutral = np.random.uniform(0.0, 0.2)
        sentiment_positive = np.random.uniform(0.7, 1.0)
    else:  # inquiry, clarification
        sentiment_negative = np.random.uniform(0.1, 0.4)
        sentiment_neutral = np.random.uniform(0.3, 0.7)
        sentiment_positive = np.random.uniform(0.1, 0.4)
    
    # Generate user message
    template = random.choice(MESSAGE_TEMPLATES[intent])
    user_message = fill_placeholders(template)
    
    return {
        "user_message": user_message,
        "anger_level": round(anger, 3),
        "frustration_level": round(frustration, 3),
        "anxiety_level": round(anxiety, 3),
        "calm_level": round(calm, 3),
        "sentiment_negative": round(sentiment_negative, 3),
        "sentiment_neutral": round(sentiment_neutral, 3),
        "sentiment_positive": round(sentiment_positive, 3),
        "intent": intent,
        "urgency": urgency,
    }


# ============================================================
# LABEL GENERATION (AGENT BEHAVIOR)
# ============================================================

def determine_agent_emotion(features: Dict) -> str:
    """
    Determine the complementary agent emotion based on user's emotional state.
    Agent emotion COMPLEMENTS (not mirrors) the user emotion.
    """
    anger = features["anger_level"]
    frustration = features["frustration_level"]
    anxiety = features["anxiety_level"]
    calm = features["calm_level"]
    intent = features["intent"]
    urgency = features["urgency"]
    
    # Priority-based complementary emotion selection
    
    # Very high anger → sad or empathetic
    if anger > 0.7:
        return random.choice(["empathetic", "sad"])
    
    # High frustration → empathetic
    if frustration > 0.6:
        return "empathetic"
    
    # High urgency (impatient) → calm
    if urgency == "high" and calm < 0.4:
        return "calm"
    
    # High anxiety → calm or reassuring
    if anxiety > 0.5:
        return random.choice(["calm", "reassuring"])
    
    # Confused (clarification intent with anxiety) → reassuring
    if intent == "clarification" and anxiety > 0.3:
        return "reassuring"
    
    # Happy/Praise → friendly
    if intent == "praise" or features["sentiment_positive"] > 0.6:
        return "friendly"
    
    # Neutral state → neutral
    if calm > 0.6 and anger < 0.2 and frustration < 0.2:
        return "neutral"
    
    # Default
    return "neutral"


def generate_labels(features: Dict) -> Dict:
    """
    Generate all agent behavior labels based on input features.
    Applies the rules defined in the specification.
    """
    anger = features["anger_level"]
    frustration = features["frustration_level"]
    anxiety = features["anxiety_level"]
    intent = features["intent"]
    urgency = features["urgency"]
    
    # ---- agent_emotion (Complementary) ----
    agent_emotion = determine_agent_emotion(features)
    
    # ---- tone ----
    # High anger or frustration → calm tone
    if anger > 0.6 or frustration > 0.6:
        tone = "calm"
    elif features["sentiment_positive"] > 0.6:
        tone = "friendly"
    else:
        tone = "neutral"
    
    # ---- empathy_level ----
    # High anger or frustration → high empathy
    if anger > 0.6 or frustration > 0.6:
        empathy_level = "high"
    elif anger > 0.3 or frustration > 0.3:
        empathy_level = "medium"
    else:
        empathy_level = "low"
    
    # ---- verbosity ----
    # High urgency → low verbosity
    if urgency == "high":
        verbosity = "low"
    elif urgency == "medium":
        verbosity = "medium"
    else:
        verbosity = "high"
    
    # ---- formality ----
    # Complaint or escalation → high formality
    if intent in ["complaint", "escalation_request"]:
        formality = "high"
    else:
        formality = "low"
    
    # ---- assertiveness ----
    # Based on anger level and escalation requests
    if anger > 0.8 or intent == "escalation_request":
        assertiveness = "high"
    elif anger > 0.4 or intent == "complaint":
        assertiveness = "medium"
    else:
        assertiveness = "low"
    
    # ---- apology_required ----
    # Complaint or escalation → apology required
    apology_required = intent in ["complaint", "escalation_request"]
    
    # ---- reassurance_level ----
    # High anxiety → high reassurance
    if anxiety > 0.5:
        reassurance_level = "high"
    elif anxiety > 0.3:
        reassurance_level = "medium"
    else:
        reassurance_level = "low"
    
    # ---- solution_speed ----
    # High urgency → fast solution
    if urgency == "high":
        solution_speed = "fast"
    elif urgency == "low":
        solution_speed = "slow"
    else:
        solution_speed = "normal"
    
    # ---- boundary_setting ----
    # Very high anger → boundary setting true
    boundary_setting = anger > 0.8
    
    # ---- explanation_depth ----
    # Clarification intent → high explanation depth
    if intent == "clarification":
        explanation_depth = "high"
    else:
        explanation_depth = "low"
    
    return {
        "agent_emotion": agent_emotion,
        "tone": tone,
        "empathy_level": empathy_level,
        "verbosity": verbosity,
        "formality": formality,
        "assertiveness": assertiveness,
        "apology_required": apology_required,
        "reassurance_level": reassurance_level,
        "solution_speed": solution_speed,
        "boundary_setting": boundary_setting,
        "explanation_depth": explanation_depth,
    }


# ============================================================
# DATASET GENERATION
# ============================================================

def generate_dataset(n_samples: int = 500) -> pd.DataFrame:
    """
    Generate the complete synthetic dataset with n_samples datapoints.
    
    Args:
        n_samples: Number of samples to generate (default 500)
    
    Returns:
        DataFrame with input features and output labels
    """
    data = []
    
    for _ in range(n_samples):
        # Generate input features
        features = generate_input_features()
        
        # Generate labels based on features
        labels = generate_labels(features)
        
        # Combine into single row
        row = {**features, **labels}
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Define column order
    input_cols = [
        "user_message", "anger_level", "frustration_level", "anxiety_level", "calm_level",
        "sentiment_negative", "sentiment_neutral", "sentiment_positive", "intent", "urgency"
    ]
    output_cols = [
        "agent_emotion", "tone", "empathy_level", "verbosity", "formality", "assertiveness",
        "apology_required", "reassurance_level", "solution_speed", "boundary_setting", "explanation_depth"
    ]
    
    df = df[input_cols + output_cols]
    
    return df


def save_dataset(df: pd.DataFrame, filepath: str) -> None:
    """Save dataset to CSV file"""
    df.to_csv(filepath, index=False)
    print(f"Dataset saved to {filepath}")
    print(f"Shape: {df.shape}")


if __name__ == "__main__":
    # Generate and save dataset
    import os
    
    # Create data directory if needed
    os.makedirs("data", exist_ok=True)
    
    # Generate dataset
    print("Generating 500 synthetic samples...")
    df = generate_dataset(500)
    
    # Save to CSV
    save_dataset(df, "data/synthetic_dataset.csv")
    
    # Show sample
    print("\nSample data (first 5 rows):")
    print(df.head())
    
    # Show label distribution
    print("\n--- Label Distributions ---")
    for col in ["agent_emotion", "tone", "empathy_level", "intent", "urgency"]:
        print(f"\n{col}:")
        print(df[col].value_counts())
