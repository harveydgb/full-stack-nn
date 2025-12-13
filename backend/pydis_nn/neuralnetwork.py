"""
Neural network module for 5D data interpolation.

This module provides a lightweight, configurable neural network implementation
using TensorFlow/Keras for regression tasks on 5-dimensional datasets.
"""

import numpy as np
import tensorflow as tf
from typing import List, Optional


class NeuralNetwork:
    """
    Configurable neural network for 5D regression tasks.
    
    Uses TensorFlow/Keras with a simple feedforward architecture.
    Designed to train quickly on CPU (<1 minute for 10K samples).
    
    Attributes:
        hidden_sizes: List of hidden layer sizes (default: [64, 32, 16])
        learning_rate: Learning rate for Adam optimizer (default: 0.001)
        max_iter: Maximum number of training epochs (default: 1000)
        random_state: Random seed for reproducibility (optional)
        model: The compiled Keras model
    """
    
    def __init__(
        self,
        hidden_sizes: List[int] = [64, 32, 16],
        learning_rate: float = 0.001,
        max_iter: int = 1000,
        random_state: int = 42
    ):
        """
        Initialize the neural network.
        
        Args:
            hidden_sizes: List of neuron counts for each hidden layer.
                         Default [64, 32, 16] gives 3 hidden layers.
            learning_rate: Learning rate for the Adam optimizer.
            max_iter: Maximum number of training epochs.
            random_state: Random seed for reproducibility. Sets both
                         TensorFlow and NumPy random seeds if provided.
        """
        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state
        self.model = None
        
        # Set random seeds for reproducibility
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
        
        # Build the model
        self._build_model()
    
    def _build_model(self):
        """Build the Keras model architecture."""
        model = tf.keras.Sequential()
        
        # Input layer - 5 features for 5D dataset
        model.add(tf.keras.layers.Input(shape=(5,)))
        
        # Hidden layers with ReLU activation
        for size in self.hidden_sizes:
            model.add(tf.keras.layers.Dense(size, activation='relu'))
        
        # Output layer - single neuron for regression, linear activation
        model.add(tf.keras.layers.Dense(1, activation='linear'))
        
        # Compile with Adam optimizer and MSE loss
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
    
    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> 'NeuralNetwork':
        """
        Train the neural network.
        
        Args:
            X: Training features array from data.py (already validated: shape (n_samples, 5))
            y: Training target array from data.py (already validated: shape (n_samples,))
            X_val: Optional validation features from data.py splits
            y_val: Optional validation targets from data.py splits
            
        Returns:
            self for method chaining (scikit-learn compatibility)
        """
        # Convert to float32 for TensorFlow efficiency
        # All shape/feature validations already done in data.py
        X = X.astype(np.float32)
        y = y.astype(np.float32)
        
        # Prepare validation data if provided (from data.py splits)
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val.astype(np.float32), y_val.astype(np.float32))
        
        # Train the model
        # Use verbose=0 to keep output clean, batch_size=32 for efficiency
        self.model.fit(
            X, y,
            epochs=self.max_iter,
            batch_size=32,
            verbose=0,
            validation_data=validation_data
        )
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Feature array from data.py (already validated: shape (n_samples, 5))
            
        Returns:
            Predictions array with shape (n_samples,)
            
        Raises:
            ValueError: If model hasn't been trained yet
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions. Call fit() first.")
        
        # Convert to float32 for TensorFlow
        # All validations already done in data.py
        X = X.astype(np.float32)
        
        # Make predictions
        predictions = self.model.predict(X, verbose=0)
        
        # Return as 1D array (squeeze the output dimension)
        return np.squeeze(predictions)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return the R² score (coefficient of determination).
        
        Useful for evaluating model performance.
        
        Args:
            X: Feature array from data.py (already validated: shape (n_samples, 5))
            y: True target values from data.py (already validated: shape (n_samples,))
            
        Returns:
            R² score
        """
        from sklearn.metrics import r2_score
        
        predictions = self.predict(X)
        return r2_score(y, predictions)

