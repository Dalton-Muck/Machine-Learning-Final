import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# File paths
file_paths = [
    './results/umap_reduced_data.csv',
    './results/tsne_reduced_data.csv',
    './results/pca_reduced_data.csv'
]

# Iterate over each file
for file_path in file_paths:
    print(f"Processing file: {file_path}")
    
    # Load the dataset
    dataset = pd.read_csv(file_path)
    
    # Split into features (X) and target variable (y)
    X = dataset[['Dimension 1', 'Dimension 2']]  # The first two columns
    y = dataset['Target']  # The last column
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=35)
    
    # Initialize the SVM model
    svm_model = SVC()
    
    # Train the SVM model
    svm_model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = svm_model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy for {file_path}: {accuracy * 100:.2f}%\n')