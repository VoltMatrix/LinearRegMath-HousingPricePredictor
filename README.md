# LinearRegMath-HousingPricePredictor
Linear Regression Implementation

Overview

This project implements linear regression from scratch using Python, NumPy, and Matplotlib, without relying on scikit-learn. It includes both single-variable and multi-variable linear regression, using gradient descent and the normal equation for optimization. The code is based on the provided implementation and works with two datasets:

ex1data1.txt: Single-variable dataset (e.g., city population vs. restaurant profit).
ex1data2.txt: Multi-variable dataset (e.g., house size, number of bedrooms vs. price).
The implementation includes:
Data loading and preprocessing (including feature normalization for multi-variable regression).

Gradient descent to optimize model parameters.

Cost function computation and visualization of convergence.

Visualization of the regression line and 3D cost surface.

Normal equation for direct parameter computation.

Prediction for new inputs (e.g., house price prediction).

Prerequisites

To run the code, you need the following Python libraries:
numpy
matplotlib
Install them using pip:

pip install numpy matplotlib

Datasets

The code expects two datasets in the data/ directory:

ex1data1.txt: Comma-separated file with two columns (e.g., population in 10,000s, profit in $10,000s).
ex1data2.txt: Comma-separated file with three columns (e.g., house size in sq ft, number of bedrooms, price).

Ensure these files are placed in a data/ folder relative to the script.

Usage

Place the dataset files (ex1data1.txt and ex1data2.txt) in a data/ directory.

Save the provided Python code as linear_regression.py (or another name).

Run the script in a Python environment:
python linear_regression.py

The script will:





Load and plot the single-variable dataset (ex1data1.txt).



Perform gradient descent to fit a linear model and plot the regression line.



Visualize the cost function convergence and the 3D cost surface with the optimization path.



Load the multi-variable dataset (ex1data2.txt), normalize features, and perform gradient descent.



Plot histograms of raw and normalized features.



Predict the price of a house with 1650 sq ft and 3 bedrooms using both gradient descent and the normal equation.

Key Components





Data Loading: Uses np.loadtxt to read comma-separated data and prepares input (X) and output (y) arrays.



Feature Normalization: Scales features (except the bias term) to have zero mean and unit variance for stable gradient descent.



Gradient Descent: Iteratively optimizes model parameters (theta) to minimize the mean squared error cost function.



Cost Function: Computes the mean squared error to evaluate model performance.



Normal Equation: Analytically solves for optimal parameters without iteration.



Visualizations:





Scatter plot with regression line (single-variable).



Cost function convergence plot.



3D cost surface with gradient descent path.



Histograms of raw and normalized features (multi-variable).
