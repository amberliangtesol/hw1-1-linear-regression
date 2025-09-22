"""
Advanced Linear Regression Application with Streamlit
Includes both sklearn and from-scratch implementations with comprehensive features
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linear_regression_from_scratch import (
    LinearRegressionPipeline, 
    DataPreparation, 
    LinearRegressionScratch,
    ModelEvaluation
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import io

st.set_page_config(page_title="Advanced Linear Regression", layout="wide", page_icon="📈")

st.title("📈 Advanced Linear Regression Implementation")
st.markdown("Complete implementation with data preparation, from-scratch model, and evaluation")
st.markdown("---")

# Sidebar for configuration
st.sidebar.header("🎛️ Configuration")

# Choose mode
mode = st.sidebar.radio(
    "Select Mode",
    ["Interactive Visualization", "From Scratch Implementation", "Upload Dataset"]
)

if mode == "Interactive Visualization":
    st.header("🎨 Interactive Linear Regression Visualization")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_points = st.slider("Number of points", 100, 1000, 300, 50)
    with col2:
        coefficient_a = st.slider("Coefficient (a)", -10.0, 10.0, 2.0, 0.1)
    with col3:
        noise_var = st.slider("Noise Variance", 0, 1000, 100, 10)
    
    intercept_b = st.slider("Intercept (b)", -50.0, 50.0, 10.0, 1.0)
    
    # Generate data
    np.random.seed(st.session_state.get('seed', 42))
    X = np.random.uniform(-50, 50, n_points).reshape(-1, 1)
    noise = np.random.normal(0, np.sqrt(noise_var), n_points)
    y = coefficient_a * X.squeeze() + intercept_b + noise
    
    # Fit model
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    # Calculate metrics
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    # Visualization
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(X, y, alpha=0.5, label='Data Points')
        ax.plot(X, y_pred, 'r-', linewidth=2, 
                label=f'y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Linear Regression Visualization')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        st.metric("MSE", f"{mse:.4f}")
        st.metric("R² Score", f"{r2:.4f}")
        st.info(f"**True Equation:**\ny = {coefficient_a}x + {intercept_b} + noise")
        st.success(f"**Fitted Equation:**\ny = {model.coef_[0]:.2f}x + {model.intercept_:.2f}")

elif mode == "From Scratch Implementation":
    st.header("🔧 Linear Regression From Scratch")
    
    # Parameters
    col1, col2 = st.columns(2)
    with col1:
        learning_rate = st.number_input("Learning Rate", 0.001, 1.0, 0.01, 0.001)
        n_iterations = st.number_input("Iterations", 100, 5000, 1000, 100)
    with col2:
        n_samples = st.number_input("Number of Samples", 100, 5000, 1000, 100)
        n_features = st.number_input("Number of Features", 1, 10, 3, 1)
    
    test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
    scale_features = st.checkbox("Scale Features", value=True)
    
    if st.button("🚀 Train Model", type="primary"):
        with st.spinner("Training model..."):
            # Generate synthetic data
            np.random.seed(42)
            X = np.random.randn(n_samples, n_features)
            
            # Create true weights
            true_weights = np.random.randn(n_features) * 3
            true_bias = np.random.randn() * 5
            noise = np.random.randn(n_samples) * 0.5
            y = np.dot(X, true_weights) + true_bias + noise
            
            # Combine data
            data = np.column_stack([X, y])
            
            # Run pipeline
            pipeline = LinearRegressionPipeline(learning_rate, n_iterations)
            results = pipeline.run_pipeline(data, scale=scale_features, 
                                          test_size=test_size, verbose=False)
            
            # Display results
            st.success("✅ Model trained successfully!")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Train MSE", f"{results['train_mse']:.4f}")
            with col2:
                st.metric("Test MSE", f"{results['test_mse']:.4f}")
            with col3:
                st.metric("Train R²", f"{results['train_r2']:.4f}")
            with col4:
                st.metric("Test R²", f"{results['test_r2']:.4f}")
            
            # Plot convergence
            st.subheader("📉 Convergence Plot")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Cost history
            ax1.plot(results['model'].cost_history)
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Cost')
            ax1.set_title('Training Cost Over Iterations')
            ax1.grid(True, alpha=0.3)
            
            # Predictions vs Actual
            ax2.scatter(results['y_test'], results['test_pred'], alpha=0.5)
            ax2.plot([results['y_test'].min(), results['y_test'].max()],
                    [results['y_test'].min(), results['y_test'].max()],
                    'r--', lw=2)
            ax2.set_xlabel('Actual Values')
            ax2.set_ylabel('Predicted Values')
            ax2.set_title('Predictions vs Actual (Test Set)')
            ax2.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            
            # Display parameters
            st.subheader("🎯 Model Parameters")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Learned Parameters:**")
                params_df = pd.DataFrame({
                    'Feature': [f'Feature {i+1}' for i in range(n_features)] + ['Bias'],
                    'Weight': list(results['model'].weights) + [results['model'].bias]
                })
                st.dataframe(params_df)
            
            with col2:
                st.write("**True Parameters:**")
                true_params_df = pd.DataFrame({
                    'Feature': [f'Feature {i+1}' for i in range(n_features)] + ['Bias'],
                    'Weight': list(true_weights) + [true_bias]
                })
                st.dataframe(true_params_df)

else:  # Upload Dataset mode
    st.header("📁 Upload Your Dataset")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Load data
        df = pd.read_csv(uploaded_file)
        st.write("**Dataset Preview:**")
        st.dataframe(df.head())
        
        st.write(f"**Shape:** {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Select target column
        target_col = st.selectbox("Select Target Column", df.columns.tolist())
        feature_cols = [col for col in df.columns if col != target_col]
        
        col1, col2 = st.columns(2)
        with col1:
            learning_rate = st.number_input("Learning Rate", 0.001, 1.0, 0.01, 0.001)
            n_iterations = st.number_input("Iterations", 100, 5000, 1000, 100)
        with col2:
            test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
            scale_features = st.checkbox("Scale Features", value=True)
        
        if st.button("🚀 Train Model on Your Data", type="primary"):
            with st.spinner("Training model on your data..."):
                # Prepare data
                X = df[feature_cols].values
                y = df[target_col].values
                
                # Check for missing values
                if np.isnan(X).any() or np.isnan(y).any():
                    st.warning("⚠️ Missing values detected. Filling with mean values...")
                
                # Create pipeline and train
                data = np.column_stack([X, y])
                pipeline = LinearRegressionPipeline(learning_rate, n_iterations)
                results = pipeline.run_pipeline(data, scale=scale_features,
                                              test_size=test_size, verbose=False)
                
                # Display results
                st.success("✅ Model trained successfully!")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Train MSE", f"{results['train_mse']:.4f}")
                with col2:
                    st.metric("Test MSE", f"{results['test_mse']:.4f}")
                with col3:
                    st.metric("Train R²", f"{results['train_r2']:.4f}")
                with col4:
                    st.metric("Test R²", f"{results['test_r2']:.4f}")
                
                # Visualizations
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                
                # Cost history
                axes[0, 0].plot(results['model'].cost_history)
                axes[0, 0].set_xlabel('Iteration')
                axes[0, 0].set_ylabel('Cost')
                axes[0, 0].set_title('Training Cost Over Iterations')
                axes[0, 0].grid(True, alpha=0.3)
                
                # Predictions vs Actual (Train)
                axes[0, 1].scatter(results['y_train'], results['train_pred'], alpha=0.5)
                axes[0, 1].plot([results['y_train'].min(), results['y_train'].max()],
                              [results['y_train'].min(), results['y_train'].max()],
                              'r--', lw=2)
                axes[0, 1].set_xlabel('Actual Values')
                axes[0, 1].set_ylabel('Predicted Values')
                axes[0, 1].set_title('Training Set: Predictions vs Actual')
                axes[0, 1].grid(True, alpha=0.3)
                
                # Predictions vs Actual (Test)
                axes[1, 0].scatter(results['y_test'], results['test_pred'], alpha=0.5)
                axes[1, 0].plot([results['y_test'].min(), results['y_test'].max()],
                              [results['y_test'].min(), results['y_test'].max()],
                              'r--', lw=2)
                axes[1, 0].set_xlabel('Actual Values')
                axes[1, 0].set_ylabel('Predicted Values')
                axes[1, 0].set_title('Test Set: Predictions vs Actual')
                axes[1, 0].grid(True, alpha=0.3)
                
                # Residuals
                residuals = results['y_test'] - results['test_pred']
                axes[1, 1].scatter(results['test_pred'], residuals, alpha=0.5)
                axes[1, 1].axhline(y=0, color='r', linestyle='--')
                axes[1, 1].set_xlabel('Predicted Values')
                axes[1, 1].set_ylabel('Residuals')
                axes[1, 1].set_title('Residual Plot')
                axes[1, 1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Feature importance
                st.subheader("📊 Feature Importance")
                importance_df = pd.DataFrame({
                    'Feature': feature_cols,
                    'Weight': results['model'].weights
                })
                importance_df = importance_df.sort_values('Weight', key=abs, ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                colors = ['green' if x > 0 else 'red' for x in importance_df['Weight']]
                ax.barh(importance_df['Feature'], importance_df['Weight'], color=colors)
                ax.set_xlabel('Weight')
                ax.set_title('Feature Weights')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

# Add information section
with st.sidebar:
    st.markdown("---")
    st.subheader("ℹ️ Information")
    st.markdown("""
    This application provides:
    
    1. **Interactive Visualization**: 
       - Real-time linear regression with adjustable parameters
       - Visual feedback on model performance
    
    2. **From Scratch Implementation**:
       - Complete implementation using gradient descent
       - Convergence monitoring
       - Performance metrics
    
    3. **Custom Dataset Support**:
       - Upload your own CSV files
       - Automatic feature detection
       - Comprehensive evaluation
    
    **Features Implemented:**
    - ✅ Data preparation (load, clean, split, scale)
    - ✅ Linear regression from scratch
    - ✅ Gradient descent optimization
    - ✅ Convergence monitoring
    - ✅ MSE and R² evaluation metrics
    - ✅ Prediction on new data
    - ✅ Comprehensive documentation
    """)
    
    if st.button("🔄 Reset Random Seed"):
        st.session_state['seed'] = np.random.randint(0, 10000)
        st.rerun()