import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

file_path = r"C:\Users\tm150\OneDrive\Desktop\PREDCT\crime sample.xlsx"

if os.path.exists(file_path):
    print("File exists and path is correct.")
    try:
        # Load the dataset
        load_dataset = pd.read_excel(file_path)
        print("First few rows of the dataset:")
        print(load_dataset.head())  # Show the first few rows of the dataset
        
        # Show dataset info
        print("\nDataset Info:")
        load_dataset.info()
        
        # Create a new DataFrame with specific columns
        dataset = pd.DataFrame()
        dataset['State'] = load_dataset['State']
        dataset['Year'] = load_dataset['Year']
        dataset['Total Cases'] = load_dataset['Victims_Total']
        
        # Show the new DataFrame
        print("\nProcessed DataFrame:")
        print(dataset.head())
        
        # Initialize LabelEncoder
        le = LabelEncoder()
        
        # Apply LabelEncoder to the 'State' column
        dataset['State'] = le.fit_transform(dataset['State'])
        
        # Show the DataFrame after encoding
        print("\nProcessed DataFrame with Encoded 'State':")
        print(dataset.head())
        
        # Split the data into features and target
        X = dataset.iloc[:, :-1]  # Features (all columns except the last one)
        Y = dataset.iloc[:, -1]   # Target variable (last column)
        
        # Show the features and target
        print("\nFeatures (X):")
        print(X.head())
        print("\nTarget (Y):")
        print(Y.head())
        
        # Split the data into training and test sets
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
        
        # Initialize the LinearRegression model
        regressor = LinearRegression()
        
        # Fit the model on the training data
        regressor.fit(X_train, Y_train)
        
        # Make predictions on the test data
        Y_pred = regressor.predict(X_test)
        
        # Print the predictions
        print("\nPredictions:")
        print(Y_pred)
        
        # Create a DataFrame to compare actual vs. predicted values
        comparison_df = pd.DataFrame({
            'ACTUAL': Y_test.values,
            'PREDICTED': Y_pred
        })
        
        # Show the comparison DataFrame
        print("\nComparison of Actual vs. Predicted Values:")
        print(comparison_df)
        
        # Calculate and print evaluation metrics
        mae = metrics.mean_absolute_error(Y_test, Y_pred)
        r2 = metrics.r2_score(Y_test, Y_pred)
        
        print("\nModel Evaluation Metrics:")
        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"R-squared (R2): {r2}")
        
        # Plotting
        plt.figure(figsize=(16, 8))
        
        # Scatter plot for actual vs predicted values
        plt.subplot(2, 2, 1)
        plt.scatter(Y_test, Y_pred, color='blue')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs. Predicted Values')
        plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=2)  # Diagonal line
        
        # Residual plot
        residuals = Y_test - Y_pred
        plt.subplot(2, 2, 2)
        plt.scatter(Y_pred, residuals, color='red')
        plt.axhline(y=0, color='black', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        
        # Histogram of Actual Values
        plt.subplot(2, 2, 3)
        plt.hist(Y_test, bins=30, color='skyblue', edgecolor='black')
        plt.xlabel('Actual Values')
        plt.ylabel('Frequency')
        plt.title('Histogram of Actual Values')
        
        # Histogram of Predicted Values
        plt.subplot(2, 2, 4)
        plt.hist(Y_pred, bins=30, color='salmon', edgecolor='black')
        plt.xlabel('Predicted Values')
        plt.ylabel('Frequency')
        plt.title('Histogram of Predicted Values')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
else:
    print("File does not exist or path is incorrect.")
