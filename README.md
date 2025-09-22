# Linear Regression Interactive Visualizer

This project provides an interactive Linear Regression visualization tool using Streamlit.

## 📁 Project Structure

```
hw1-1-linear-regression/
│
├── app.py              # Interactive visualization app
├── requirements.txt    # Project dependencies
├── TODO.md            # Project task tracking
└── README.md          # This file
```

## ✨ Features

### Interactive Visualization
- **Data Generation**: Generate n data points (100-1000) with linear relationship y = ax + b + noise
- **Adjustable Parameters**:
  - Number of data points (n)
  - Coefficient (a): -10 to 10
  - Intercept (b): -50 to 50  
  - Noise variance: 0 to 1000
- **Real-time Updates**: Instant visualization when parameters change
- **Generate New Data**: Button to create new random datasets

### Linear Regression
- **Automatic Calculation**: Uses scikit-learn for robust regression
- **Visual Representation**: Red regression line on scatter plot
- **Statistics Display**: 
  - R² score
  - Mean Squared Error (MSE)
  - Regression equation
  - True equation comparison

### Outlier Detection
- **Top 5 Outliers**: Automatically identifies points furthest from regression line
- **Visual Highlighting**: Orange markers with red borders
- **Labeled Points**: Each outlier labeled (O1-O5)
- **Distance Metrics**: Table showing outlier distances from regression line

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## 💡 How to Use

1. **Adjust Parameters**: Use the sliders in the sidebar to set:
   - Number of data points (n): 100-1000
   - Coefficient (a): -10 to 10 for the linear relationship
   - Intercept (b): -50 to 50
   - Noise variance: 0-1000 to add randomness

2. **Generate Data**: Click the '🔄 Generate New Data' button to create a new dataset

3. **Analyze Results**: 
   - The red line shows the calculated linear regression
   - Orange points with red borders are the top 5 outliers
   - Check the statistics panel for R² score and MSE
   - Compare fitted equation with true equation

## 🌐 Deployment

The application is deployed on Streamlit Cloud and available at:
- **Live App**: https://2025hw1.streamlit.app/
- **GitHub Repository**: https://github.com/amberliangtesol/hw1-1-linear-regression

To deploy your own version:
1. Fork the repository
2. Go to https://share.streamlit.io/
3. Connect your GitHub account
4. Deploy with repository path and `app.py` as main file

## 🔧 Requirements

- Python 3.7+
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Streamlit

## 📝 Project Description

This interactive Linear Regression visualizer demonstrates:
- Real-time parameter adjustment and visualization
- Outlier detection algorithms
- Statistical metrics calculation
- The relationship between true and fitted models

Perfect for understanding how linear regression works with varying levels of noise and different parameter settings.

## 📜 License

This project is for educational purposes.

## 🙏 Acknowledgments

Created as part of HW1-1 assignment for Linear Regression visualization with Streamlit.