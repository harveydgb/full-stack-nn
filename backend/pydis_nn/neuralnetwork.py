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
        max_iter: Maximum number of training epochs (default: 300)
        random_state: Random seed for reproducibility (optional)
        model: The compiled Keras model
    """
    
    def __init__(
        self,
        hidden_sizes: List[int] = [64, 32, 16],
        learning_rate: float = 0.001,
        max_iter: int = 300,
        random_state: int = 42,
        early_stopping_patience: int = 50
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
            early_stopping_patience: Number of epochs with no improvement after
                                   which training will be stopped. Only used if
                                   validation data is provided. Default: 50.
        """
        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state
        self.early_stopping_patience = early_stopping_patience
        self.model = None
        self._is_trained = False
        
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
        y_val: Optional[np.ndarray] = None,
        return_history: bool = False
    ):
        """
        Train the neural network.
        
        Args:
            X: Training features (n_samples, 5)
            y: Training targets (n_samples,)
            X_val: Optional validation features
            y_val: Optional validation targets
            return_history: If True, return training history along with self
            
        Returns:
            self (if return_history=False) or tuple of (self, history_dict) (if return_history=True)
        """
        # Convert to float32 for TensorFlow
        X = X.astype(np.float32)
        y = y.astype(np.float32)
        
        # Prepare validation data if provided
        validation_data = None
        callbacks = []
        loss_history_list = []
        
        if X_val is not None and y_val is not None:
            validation_data = (X_val.astype(np.float32), y_val.astype(np.float32))
            # Early stopping callback for validation data
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.early_stopping_patience,
                restore_best_weights=True,
                verbose=0
            )
            callbacks.append(early_stopping)
        
        # Loss history callback
        if return_history:
            class LossHistory(tf.keras.callbacks.Callback):
                def __init__(self, history_list):
                    super().__init__()
                    self.history_list = history_list
                
                def on_epoch_end(self, epoch, logs=None):
                    if logs is not None:
                        self.history_list.append({
                            'epoch': epoch,
                            'loss': float(logs.get('loss', 0)),
                            'val_loss': float(logs.get('val_loss', logs.get('loss', 0)))
                        })
            
            callbacks.append(LossHistory(loss_history_list))
        
        # Train model with batch_size=32, verbose=0
        self.model.fit(
            X, y,
            epochs=self.max_iter,
            batch_size=32,
            verbose=0,
            validation_data=validation_data,
            callbacks=callbacks
        )
        
        self._is_trained = True
        
        if return_history:
            return self, loss_history_list
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Feature array (n_samples, 5)
            
        Returns:
            Predictions array with shape (n_samples,)
            
        Raises:
            ValueError: If model hasn't been trained yet
        """
        if self.model is None or not self._is_trained:
            raise ValueError("Model must be trained before making predictions. Call fit() first.")
        
        # Convert to float32 for TensorFlow
        X = X.astype(np.float32)
        
        # Make predictions
        predictions = self.model.predict(X, verbose=0)
        
        # Return as 1D array (squeeze the output dimension)
        predictions = np.squeeze(predictions)
        
        # Ensure we always return at least 1D array (not scalar) for consistency
        if predictions.ndim == 0:
            predictions = np.array([predictions])
        
        return predictions
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return the R² score (coefficient of determination).
        
        Useful for evaluating model performance.
        
        Args:
            X: Feature array (n_samples, 5)
            y: True target values (n_samples,)
            
        Returns:
            R² score
        """
        from sklearn.metrics import r2_score
        
        predictions = self.predict(X)
        return r2_score(y, predictions)
    
    def evaluate_all(self, X_train: np.ndarray, y_train: np.ndarray,
                     X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
                     X_test: Optional[np.ndarray] = None, y_test: Optional[np.ndarray] = None) -> dict:
        """
        Evaluate model performance on train, validation, and test sets.
        
        Computes R² scores and MSE for all provided datasets.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Optional validation features
            y_val: Optional validation targets
            X_test: Optional test features
            y_test: Optional test targets
            
        Returns:
            Dictionary with metrics for each dataset provided:
            - 'train_r2', 'train_mse' (always present)
            - 'val_r2', 'val_mse' (if X_val/y_val provided)
            - 'test_r2', 'test_mse' (if X_test/y_test provided)
        """
        from sklearn.metrics import mean_squared_error
        
        metrics = {}
        
        # Train metrics
        train_pred = self.predict(X_train)
        metrics['train_r2'] = float(self.score(X_train, y_train))
        metrics['train_mse'] = float(mean_squared_error(y_train, train_pred))
        
        # Validation metrics
        if X_val is not None and y_val is not None:
            val_pred = self.predict(X_val)
            metrics['val_r2'] = float(self.score(X_val, y_val))
            metrics['val_mse'] = float(mean_squared_error(y_val, val_pred))
        
        # Test metrics
        if X_test is not None and y_test is not None:
            test_pred = self.predict(X_test)
            metrics['test_r2'] = float(self.score(X_test, y_test))
            metrics['test_mse'] = float(mean_squared_error(y_test, test_pred))
        
        return metrics

