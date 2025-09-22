# Linear Regression Implementation - Complete Package

This project provides a comprehensive implementation of Linear Regression, including both a from-scratch implementation and an interactive visualization tool.

## 📁 Project Structure

```
hw1-1-linear-regression/
│
├── app.py                          # Interactive visualization app (original requirement)
├── app_advanced.py                 # Advanced app with all features
├── linear_regression_from_scratch.py  # Complete from-scratch implementation
├── requirements.txt                # Project dependencies
└── README.md                       # This file
```

## ✅ Completed To-Do Items

### 1. Data Preparation ✅
- **Load dataset**: Support for CSV, NumPy arrays, and Pandas DataFrames
- **Handle missing values**: Strategies include mean, median, or drop
- **Split data**: Train/test splitting with configurable ratio
- **Feature scaling**: StandardScaler implementation

### 2. Model Implementation ✅
- **From-scratch implementation**: Complete LinearRegressionScratch class
- **Initialize weights and bias**: Automatic initialization
- **Hypothesis function**: h(x) = wx + b implementation
- **Cost function**: Mean Squared Error (MSE) implementation
- **Gradient Descent**: Full implementation with:
  - Gradient calculation
  - Parameter updates
  - Learning rate control

### 3. Training ✅
- **Model training**: Fit method with gradient descent
- **Convergence monitoring**: Cost history tracking and visualization

### 4. Evaluation ✅
- **Predictions**: Support for test set predictions
- **Evaluation metrics**:
  - Mean Squared Error (MSE) ✅
  - R-squared (R²) Score ✅
- **Visualizations**: Predictions vs actual values plots

### 5. Prediction ✅
- **New data predictions**: `predict_new()` function for unseen data
- **Scaling support**: Automatic scaling if training data was scaled

### 6. Documentation ✅
- **Code documentation**: Comprehensive docstrings for all classes and methods
- **Usage examples**: Included in the code
- **This README**: Complete project documentation

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Running the Applications

#### 1. Original Interactive Visualizer
```bash
streamlit run app.py
```
Features:
- Interactive parameter adjustment (n, a, variance)
- Real-time visualization
- Outlier detection (top 5)
- Statistics display

#### 2. Advanced Application
```bash
streamlit run app_advanced.py
```
Features:
- Three modes: Interactive, From Scratch, Upload Dataset
- Complete from-scratch implementation
- Convergence monitoring
- Feature importance analysis
- Support for custom datasets

#### 3. Run From-Scratch Implementation Directly
```python
python linear_regression_from_scratch.py
```

## 📊 Usage Examples

### Using the Pipeline

```python
from linear_regression_from_scratch import LinearRegressionPipeline
import numpy as np

# Generate data
X = np.random.randn(1000, 3)
y = 2*X[:, 0] - 1.5*X[:, 1] + 0.5*X[:, 2] + 3 + np.random.randn(1000)*0.5
data = np.column_stack([X, y])

# Create and run pipeline
pipeline = LinearRegressionPipeline(learning_rate=0.01, n_iterations=1000)
results = pipeline.run_pipeline(data, scale=True, test_size=0.2)

# Make predictions on new data
X_new = np.random.randn(10, 3)
predictions = pipeline.predict_new(X_new)
```

### Using Individual Components

```python
from linear_regression_from_scratch import DataPreparation, LinearRegressionScratch, ModelEvaluation

# Data preparation
data_prep = DataPreparation()
X, y = data_prep.load_data('your_data.csv')
X, y = data_prep.handle_missing_values(X, y, strategy='mean')
X_train, X_test, y_train, y_test = data_prep.split_data(X, y)
X_train, X_test = data_prep.scale_features(X_train, X_test)

# Model training
model = LinearRegressionScratch(learning_rate=0.01, n_iterations=1000)
model.fit(X_train, y_train, verbose=True)

# Evaluation
evaluator = ModelEvaluation()
y_pred = model.predict(X_test)
mse = evaluator.mean_squared_error(y_test, y_pred)
r2 = evaluator.r2_score(y_test, y_pred)
```

## 📈 Features

### DataPreparation Class
- Load data from multiple sources
- Handle missing values (mean/median/drop)
- Train/test splitting
- Feature scaling with StandardScaler

### LinearRegressionScratch Class
- Initialize parameters (weights and bias)
- Hypothesis function implementation
- MSE cost function
- Gradient descent optimization
- Convergence tracking

### ModelEvaluation Class
- Mean Squared Error (MSE)
- R-squared score
- Visualization tools

### LinearRegressionPipeline Class
- Complete end-to-end pipeline
- Automatic preprocessing
- Model training and evaluation
- Support for new predictions

## 📊 Visualizations

The implementation provides several visualization options:

1. **Convergence Plot**: Shows cost reduction over iterations
2. **Predictions vs Actual**: Scatter plot comparing predictions to true values
3. **Residual Plot**: Shows prediction errors
4. **Feature Importance**: Bar chart of feature weights

## 🎯 Key Algorithms

### Gradient Descent
```
for iteration in range(n_iterations):
    1. Calculate predictions: ŷ = Xw + b
    2. Calculate error: e = ŷ - y
    3. Calculate gradients:
       - dw = (1/m) * X^T * e
       - db = (1/m) * sum(e)
    4. Update parameters:
       - w = w - α * dw
       - b = b - α * db
```

### Cost Function (MSE)
```
J(w, b) = (1/2m) * Σ(ŷᵢ - yᵢ)²
```

### R² Score
```
R² = 1 - (SS_res / SS_tot)
where:
- SS_res = Σ(yᵢ - ŷᵢ)²
- SS_tot = Σ(yᵢ - ȳ)²
```

## 🔧 Requirements

- Python 3.7+
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Streamlit

## 📝 Notes

- The from-scratch implementation uses pure NumPy for all computations
- The pipeline supports both scaled and unscaled features
- Convergence monitoring helps identify optimal iteration counts
- The implementation includes proper train/test splitting to evaluate generalization

## 🎓 Educational Value

This implementation is ideal for:
- Understanding linear regression fundamentals
- Learning gradient descent optimization
- Practicing data preprocessing techniques
- Comparing custom implementations with sklearn
- Visualizing machine learning concepts

## 📜 License

This project is for educational purposes.

## 🙏 Acknowledgments

Created as part of HW1-1 assignment for understanding Linear Regression implementation from scratch.