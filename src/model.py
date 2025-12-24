"""
Multi-Output Neural Network Model for Persona Prediction
Predicts all agent behavior fields from processed features.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Tuple
import os


class MultiOutputModel(nn.Module):
    """
    Multi-output neural network with shared hidden layers
    and separate output heads for each agent behavior field.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dims: Dict[str, int],
        dropout: float = 0.3
    ):
        """
        Args:
            input_dim: Size of input feature vector
            hidden_dims: List of hidden layer dimensions
            output_dims: Dict mapping output names to number of classes
            dropout: Dropout probability
        """
        super().__init__()
        
        # Store architecture config for saving/loading
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.output_dims = output_dims
        self.output_names = list(output_dims.keys())
        
        # Build shared layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Build output heads (one per output field)
        self.output_heads = nn.ModuleDict()
        for name, n_classes in output_dims.items():
            self.output_heads[name] = nn.Linear(prev_dim, n_classes)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning logits for each output.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Dict mapping output names to logits tensors
        """
        # Shared representation
        shared = self.shared_layers(x)
        
        # Output heads
        outputs = {}
        for name, head in self.output_heads.items():
            outputs[name] = head(shared)
        
        return outputs
    
    def predict(self, x: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        Get predicted class indices for each output.
        
        Args:
            x: Input tensor
            
        Returns:
            Dict mapping output names to predicted class indices
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            predictions = {}
            for name, logit in logits.items():
                predictions[name] = torch.argmax(logit, dim=1).cpu().numpy()
        return predictions


class ModelTrainer:
    """
    Trainer for the multi-output model.
    Handles training, evaluation, and saving/loading.
    """
    
    def __init__(
        self,
        model: MultiOutputModel,
        learning_rate: float = 0.001,
        device: str = None
    ):
        """
        Args:
            model: The MultiOutputModel to train
            learning_rate: Learning rate for optimizer
            device: Device to use ('cuda' or 'cpu')
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Loss functions: CrossEntropy for all (binary outputs also have 2 classes)
        self.criterion = nn.CrossEntropyLoss()
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        labels_dict: Dict[str, torch.Tensor]
    ) -> float:
        """
        Train for one epoch.
        
        Args:
            dataloader: DataLoader for training data
            labels_dict: Dict mapping output names to label tensors
            
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        for batch_idx, (X_batch,) in enumerate(dataloader):
            X_batch = X_batch.to(self.device)
            
            # Get batch labels
            batch_labels = {}
            batch_start = batch_idx * dataloader.batch_size
            batch_end = min(batch_start + dataloader.batch_size, len(labels_dict[self.model.output_names[0]]))
            
            for name in self.model.output_names:
                batch_labels[name] = labels_dict[name][batch_start:batch_end].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(X_batch)
            
            # Compute combined loss (sum of all output losses)
            loss = 0.0
            for name in self.model.output_names:
                loss += self.criterion(outputs[name], batch_labels[name])
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches
    
    def evaluate(
        self,
        X: torch.Tensor,
        labels_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Evaluate accuracy for each output field.
        
        Args:
            X: Feature tensor
            labels_dict: Dict mapping output names to label tensors
            
        Returns:
            Dict mapping output names to accuracy scores
        """
        self.model.eval()
        X = X.to(self.device)
        
        predictions = self.model.predict(X)
        
        accuracies = {}
        for name in self.model.output_names:
            y_true = labels_dict[name].cpu().numpy()
            y_pred = predictions[name]
            accuracies[name] = np.mean(y_true == y_pred)
        
        return accuracies
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: Dict[str, np.ndarray],
        epochs: int = 50,
        batch_size: int = 32,
        verbose: bool = True
    ) -> List[float]:
        """
        Train the model.
        
        Args:
            X_train: Training features array
            y_train: Dict mapping output names to training labels
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: Whether to print progress
            
        Returns:
            List of training losses per epoch
        """
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_train)
        labels_tensors = {
            name: torch.LongTensor(labels) for name, labels in y_train.items()
        }
        
        # Create dataloader
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        losses = []
        for epoch in range(epochs):
            loss = self.train_epoch(dataloader, labels_tensors)
            losses.append(loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")
        
        return losses
    
    def save(self, filepath: str) -> None:
        """Save model state dict and architecture config"""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "output_names": self.model.output_names,
            "input_dim": self.model.input_dim,
            "hidden_dims": self.model.hidden_dims,
            "output_dims": self.model.output_dims,
            "dropout": self.model.dropout,
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load model state dict"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Model loaded from {filepath}")


def create_model(
    input_dim: int,
    output_dims: Dict[str, int],
    hidden_dims: List[int] = [128, 64],
    dropout: float = 0.3
) -> MultiOutputModel:
    """
    Factory function to create a MultiOutputModel.
    
    Args:
        input_dim: Size of input features
        output_dims: Dict mapping output names to number of classes
        hidden_dims: List of hidden layer sizes
        dropout: Dropout probability
        
    Returns:
        Configured MultiOutputModel instance
    """
    return MultiOutputModel(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dims=output_dims,
        dropout=dropout
    )


if __name__ == "__main__":
    # Test the model
    print("Testing Multi-Output Model...")
    
    # Dummy data
    input_dim = 115  # 7 numeric + 8 categorical + 100 tfidf
    output_dims = {
        "agent_emotion": 6,
        "tone": 3,
        "empathy_level": 3,
        "verbosity": 3,
        "formality": 2,
        "assertiveness": 3,
        "apology_required": 2,
        "reassurance_level": 3,
        "solution_speed": 3,
        "boundary_setting": 2,
        "explanation_depth": 2,
    }
    
    # Create model
    model = create_model(input_dim, output_dims)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    X = torch.randn(32, input_dim)
    outputs = model(X)
    
    print("\nOutput shapes:")
    for name, out in outputs.items():
        print(f"  {name}: {out.shape}")
