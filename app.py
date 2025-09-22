import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd

st.set_page_config(page_title="Linear Regression Visualizer", layout="wide")

st.title("🔢 Interactive Linear Regression Visualizer")
st.markdown("---")

# Sidebar for parameter controls
st.sidebar.header("⚙️ Parameters")

# User input for number of data points
n_points = st.sidebar.slider(
    "Number of data points (n)", 
    min_value=100, 
    max_value=1000, 
    value=300,
    step=50
)

# User input for coefficient 'a'
coefficient_a = st.sidebar.slider(
    "Coefficient (a) for y = ax + b + noise", 
    min_value=-10.0, 
    max_value=10.0, 
    value=2.0,
    step=0.1
)

# User input for intercept 'b'
intercept_b = st.sidebar.slider(
    "Intercept (b) for y = ax + b + noise", 
    min_value=-50.0, 
    max_value=50.0, 
    value=10.0,
    step=1.0
)

# User input for noise variance
noise_variance = st.sidebar.slider(
    "Noise Variance (σ²)", 
    min_value=0, 
    max_value=1000, 
    value=100,
    step=10
)

# Generate data button
generate_button = st.sidebar.button("🔄 Generate New Data", type="primary", use_container_width=True)

@st.cache_data
def generate_data(n, a, b, var, seed=None):
    """Generate synthetic data with linear relationship and noise."""
    if seed is not None:
        np.random.seed(seed)
    
    # Generate x values
    x = np.random.uniform(-50, 50, n)
    
    # Generate noise
    noise = np.random.normal(0, np.sqrt(var), n)
    
    # Calculate y values
    y = a * x + b + noise
    
    return x, y

@st.cache_data
def perform_linear_regression(x, y):
    """Perform linear regression and return model and predictions."""
    # Reshape x for sklearn
    X = x.reshape(-1, 1)
    
    # Create and fit the model
    model = LinearRegression()
    model.fit(X, y)
    
    # Get predictions
    y_pred = model.predict(X)
    
    # Calculate R-squared
    r2 = model.score(X, y)
    
    return model, y_pred, r2

@st.cache_data
def find_outliers(x, y, y_pred, n_outliers=5):
    """Find the top n outliers based on distance from regression line."""
    # Calculate residuals (distances from regression line)
    residuals = np.abs(y - y_pred)
    
    # Get indices of top n outliers
    outlier_indices = np.argsort(residuals)[-n_outliers:]
    
    return outlier_indices, residuals

# Initialize session state for random seed
if 'random_seed' not in st.session_state:
    st.session_state.random_seed = 42

# Update seed when generate button is clicked
if generate_button:
    st.session_state.random_seed = np.random.randint(0, 10000)

# Generate data
x, y = generate_data(n_points, coefficient_a, intercept_b, noise_variance, st.session_state.random_seed)

# Perform linear regression
model, y_pred, r2 = perform_linear_regression(x, y)

# Find outliers
outlier_indices, residuals = find_outliers(x, y, y_pred)

# Create two columns for layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Visualization")
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot all data points
    scatter = ax.scatter(x, y, alpha=0.5, s=30, label='Data Points', color='blue')
    
    # Plot regression line
    sorted_indices = np.argsort(x)
    ax.plot(x[sorted_indices], y_pred[sorted_indices], 'r-', linewidth=2, 
            label=f'Regression Line: y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}')
    
    # Highlight outliers
    ax.scatter(x[outlier_indices], y[outlier_indices], 
              color='orange', s=100, edgecolors='red', linewidth=2,
              label='Top 5 Outliers', zorder=5)
    
    # Add labels to outliers
    for i, idx in enumerate(outlier_indices):
        ax.annotate(f'O{i+1}', 
                   (x[idx], y[idx]), 
                   xytext=(5, 5), 
                   textcoords='offset points',
                   fontsize=8,
                   color='red',
                   fontweight='bold')
    
    # Customize plot
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title(f'Linear Regression with n={n_points} points', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Display the plot
    st.pyplot(fig)

with col2:
    st.subheader("📈 Statistics")
    
    # Display regression equation
    st.info(f"**Regression Equation:**\n\ny = {model.coef_[0]:.3f}x + {model.intercept_:.3f}")
    
    # Display R-squared
    st.metric("R² Score", f"{r2:.4f}")
    
    # Display true equation
    st.success(f"**True Equation:**\n\ny = {coefficient_a:.1f}x + {intercept_b:.1f} + N(0, {noise_variance})")
    
    # Display outlier information
    st.subheader("🎯 Top 5 Outliers")
    
    outlier_data = []
    for i, idx in enumerate(outlier_indices):
        outlier_data.append({
            'Outlier': f'O{i+1}',
            'X': f'{x[idx]:.2f}',
            'Y': f'{y[idx]:.2f}',
            'Distance': f'{residuals[idx]:.2f}'
        })
    
    outlier_df = pd.DataFrame(outlier_data)
    outlier_df = outlier_df.sort_values('Distance', ascending=False)
    st.dataframe(outlier_df, use_container_width=True, hide_index=True)
    
    # Additional statistics
    st.subheader("📊 Data Statistics")
    stats_col1, stats_col2 = st.columns(2)
    
    with stats_col1:
        st.metric("Mean X", f"{np.mean(x):.2f}")
        st.metric("Mean Y", f"{np.mean(y):.2f}")
    
    with stats_col2:
        st.metric("Std X", f"{np.std(x):.2f}")
        st.metric("Std Y", f"{np.std(y):.2f}")

# Footer
st.markdown("---")
st.markdown("### 💡 How to Use")
st.markdown("""
1. **Adjust Parameters**: Use the sliders in the sidebar to set:
   - Number of data points (n)
   - Coefficient (a) for the linear relationship
   - Intercept (b) for the linear relationship
   - Noise variance to add randomness
   
2. **Generate Data**: Click the '🔄 Generate New Data' button to create a new dataset
   
3. **Analyze Results**: 
   - The red line shows the calculated linear regression
   - Orange points with red borders are the top 5 outliers
   - Check the statistics panel for detailed metrics
""")