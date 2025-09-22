"""
Linear Regression Implementation from Scratch
Complete implementation including data preparation, model training, and evaluation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional, Union
import os


class DataPreparation:
    """Handle data loading, cleaning, splitting, and scaling"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_scaled = False
        
    def load_data(self, source: Union[str, np.ndarray, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data from various sources
        
        Args:
            source: Can be a file path, numpy array, or pandas DataFrame
            
        Returns:
            X: Feature matrix
            y: Target vector
        """
        if isinstance(source, str):
            # Load from file
            if source.endswith('.csv'):
                df = pd.read_csv(source)
                # Assume last column is target
                X = df.iloc[:, :-1].values
                y = df.iloc[:, -1].values
            elif source.endswith('.npy'):
                data = np.load(source)
                X = data[:, :-1]
                y = data[:, -1]
            else:
                raise ValueError(f"Unsupported file format: {source}")
        elif isinstance(source, pd.DataFrame):
            X = source.iloc[:, :-1].values
            y = source.iloc[:, -1].values
        elif isinstance(source, np.ndarray):
            X = source[:, :-1]
            y = source[:, -1]
        else:
            raise ValueError(f"Unsupported data type: {type(source)}")
            
        return X, y
    
    def handle_missing_values(self, X: np.ndarray, y: np.ndarray, 
                            strategy: str = 'mean') -> Tuple[np.ndarray, np.ndarray]:
        """
        Handle missing values in the dataset
        
        Args:
            X: Feature matrix
            y: Target vector
            strategy: 'mean', 'median', or 'drop'
            
        Returns:
            Cleaned X and y
        """
        # Find rows with NaN values
        nan_mask_X = np.isnan(X).any(axis=1)
        nan_mask_y = np.isnan(y)
        nan_mask = nan_mask_X | nan_mask_y
        
        if strategy == 'drop':
            # Remove rows with NaN
            X = X[~nan_mask]
            y = y[~nan_mask]
        elif strategy == 'mean':
            # Fill with mean
            for i in range(X.shape[1]):
                col_mean = np.nanmean(X[:, i])
                X[np.isnan(X[:, i]), i] = col_mean
            y_mean = np.nanmean(y)
            y[np.isnan(y)] = y_mean
        elif strategy == 'median':
            # Fill with median
            for i in range(X.shape[1]):
                col_median = np.nanmedian(X[:, i])
                X[np.isnan(X[:, i]), i] = col_median
            y_median = np.nanmedian(y)
            y[np.isnan(y)] = y_median
            
        return X, y
    
    def split_data(self, X: np.ndarray, y: np.ndarray, 
                   test_size: float = 0.2, 
                   random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training and testing sets
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Proportion of data for testing
            random_state: Random seed
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    def scale_features(self, X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scale features using StandardScaler
        
        Args:
            X_train: Training features
            X_test: Testing features
            
        Returns:
            Scaled X_train and X_test
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        self.is_scaled = True
        return X_train_scaled, X_test_scaled


class LinearRegressionScratch:
    """Linear Regression implementation from scratch using Gradient Descent"""
    
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        """
        Initialize Linear Regression model
        
        Args:
            learning_rate: Learning rate for gradient descent
            n_iterations: Number of iterations for training
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []
        
    def initialize_parameters(self, n_features: int):
        """
        Initialize weights and bias
        
        Args:
            n_features: Number of features
        """
        self.weights = np.zeros(n_features)
        self.bias = 0
        
    def hypothesis(self, X: np.ndarray) -> np.ndarray:
        """
        Compute hypothesis h(x) = wx + b
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions
        """
        return np.dot(X, self.weights) + self.bias
    
    def cost_function(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate Mean Squared Error cost
        
        Args:
            X: Feature matrix
            y: True values
            
        Returns:
            MSE cost
        """
        m = len(y)
        predictions = self.hypothesis(X)
        cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
        return cost
    
    def calculate_gradients(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Calculate gradients for weights and bias
        
        Args:
            X: Feature matrix
            y: True values
            
        Returns:
            Weight gradients and bias gradient
        """
        m = len(y)
        predictions = self.hypothesis(X)
        error = predictions - y
        
        dw = (1 / m) * np.dot(X.T, error)
        db = (1 / m) * np.sum(error)
        
        return dw, db
    
    def update_parameters(self, dw: np.ndarray, db: float):
        """
        Update weights and bias using gradients
        
        Args:
            dw: Weight gradients
            db: Bias gradient
        """
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db
    
    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False):
        """
        Train the linear regression model using gradient descent
        
        Args:
            X: Training features
            y: Training targets
            verbose: Print training progress
        """
        # Initialize parameters
        self.initialize_parameters(X.shape[1])
        
        # Gradient descent
        for i in range(self.n_iterations):
            # Calculate cost
            cost = self.cost_function(X, y)
            self.cost_history.append(cost)
            
            # Calculate gradients
            dw, db = self.calculate_gradients(X, y)
            
            # Update parameters
            self.update_parameters(dw, db)
            
            # Print progress
            if verbose and (i % 100 == 0 or i == self.n_iterations - 1):
                print(f"Iteration {i}: Cost = {cost:.4f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions
        """
        return self.hypothesis(X)
    
    def plot_convergence(self):
        """Plot the cost function over iterations to monitor convergence"""
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.cost_history)), self.cost_history)
        plt.xlabel('Iteration')
        plt.ylabel('Cost (MSE)')
        plt.title('Convergence of Gradient Descent')
        plt.grid(True)
        plt.show()


class ModelEvaluation:
    """Evaluate the performance of the linear regression model"""
    
    @staticmethod
    def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Squared Error
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            MSE
        """
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate R-squared score
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            R2 score
        """
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2
    
    @staticmethod
    def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray):
        """
        Visualize predictions vs actual values
        
        Args:
            y_true: True values
            y_pred: Predicted values
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Predictions vs Actual Values')
        plt.grid(True)
        plt.show()


class LinearRegressionPipeline:
    """Complete pipeline for linear regression"""
    
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        self.data_prep = DataPreparation()
        self.model = LinearRegressionScratch(learning_rate, n_iterations)
        self.evaluator = ModelEvaluation()
        
    def run_pipeline(self, data_source: Union[str, np.ndarray, pd.DataFrame], 
                    scale: bool = False,
                    test_size: float = 0.2,
                    verbose: bool = True) -> dict:
        """
        Run complete linear regression pipeline
        
        Args:
            data_source: Data source
            scale: Whether to scale features
            test_size: Test set size
            verbose: Print progress
            
        Returns:
            Dictionary with results
        """
        # 1. Load data
        X, y = self.data_prep.load_data(data_source)
        if verbose:
            print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
        
        # 2. Handle missing values
        X, y = self.data_prep.handle_missing_values(X, y)
        
        # 3. Split data
        X_train, X_test, y_train, y_test = self.data_prep.split_data(X, y, test_size)
        if verbose:
            print(f"Train set: {X_train.shape[0]} samples")
            print(f"Test set: {X_test.shape[0]} samples")
        
        # 4. Scale features if requested
        if scale:
            X_train, X_test = self.data_prep.scale_features(X_train, X_test)
            if verbose:
                print("Features scaled")
        
        # 5. Train model
        if verbose:
            print("\nTraining model...")
        self.model.fit(X_train, y_train, verbose=verbose)
        
        # 6. Make predictions
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        # 7. Evaluate
        train_mse = self.evaluator.mean_squared_error(y_train, train_pred)
        test_mse = self.evaluator.mean_squared_error(y_test, test_pred)
        train_r2 = self.evaluator.r2_score(y_train, train_pred)
        test_r2 = self.evaluator.r2_score(y_test, test_pred)
        
        if verbose:
            print(f"\nTraining MSE: {train_mse:.4f}")
            print(f"Testing MSE: {test_mse:.4f}")
            print(f"Training R2: {train_r2:.4f}")
            print(f"Testing R2: {test_r2:.4f}")
        
        results = {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'model': self.model,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'train_pred': train_pred,
            'test_pred': test_pred
        }
        
        return results
    
    def predict_new(self, X_new: np.ndarray) -> np.ndarray:
        """
        Make predictions on completely new data
        
        Args:
            X_new: New feature matrix
            
        Returns:
            Predictions
        """
        if self.data_prep.is_scaled:
            X_new = self.data_prep.scaler.transform(X_new)
        return self.model.predict(X_new)


# Example usage and testing
if __name__ == "__main__":
    # Generate synthetic data for testing
    np.random.seed(42)
    n_samples = 1000
    n_features = 3
    
    # Create synthetic dataset
    X = np.random.randn(n_samples, n_features)
    true_weights = np.array([2.5, -1.3, 0.7])
    true_bias = 3.2
    noise = np.random.randn(n_samples) * 0.5
    y = np.dot(X, true_weights) + true_bias + noise
    
    # Combine into single array
    data = np.column_stack([X, y])
    
    print("=" * 50)
    print("Linear Regression from Scratch - Complete Pipeline")
    print("=" * 50)
    
    # Run pipeline
    pipeline = LinearRegressionPipeline(learning_rate=0.1, n_iterations=500)
    results = pipeline.run_pipeline(data, scale=True, test_size=0.2, verbose=True)
    
    # Plot convergence
    print("\nPlotting convergence...")
    pipeline.model.plot_convergence()
    
    # Plot predictions
    print("Plotting predictions vs actual...")
    pipeline.evaluator.plot_predictions(results['y_test'], results['test_pred'])
    
    # Test prediction on new data
    X_new = np.random.randn(5, n_features)
    predictions = pipeline.predict_new(X_new)
    print(f"\nPredictions on new data: {predictions}")
    
    # Display learned parameters
    print(f"\nLearned weights: {pipeline.model.weights}")
    print(f"Learned bias: {pipeline.model.bias:.4f}")
    print(f"\nTrue weights: {true_weights}")
    print(f"True bias: {true_bias}")