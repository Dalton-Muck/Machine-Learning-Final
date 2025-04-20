import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report

    
RESET = '\033[0m'
GREEN = '\033[32m'
BLUE = '\033[34m'
    
# File paths
file_paths = [
    # './results/umap_reduced_data.csv',
    './results/tsne_reduced_data.csv',
    # './results/pca_reduced_data.csv'
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
    
    
    ######################################################################################
    #Grid Search for SVM
    ######################################################################################
    # Define the parameter grid for SVM
    # param_grid = [
    # {
    #     'C': [.05, 0.1, 1],
    #     'kernel': ['poly'],
    #     'gamma': ['auto', .05, 0.1, 0.01],
    #     'degree': [2, 3, 4],  # Removed 'auto' since 'degree' requires an integer
    #     'coef0': [2.0, 3.0, 4.0],
    #     'tol': [0.0001]
    # },
    # {
    #     'C': [.1, 1],
    #     'kernel': ['sigmoid'],
    #     'gamma': ['scale', 'auto', .001, .01, .1],
    #     'coef0': [0.0, 1.0],
    #     'tol': [0.0001]
    # },
    # {
    #     'C': [.01, .1, 1],
    #     'kernel': ['rbf'],
    #     'tol': [0.0001],
    #     'gamma': ['scale', 'auto'],
    #     'cache_size': [500],
    #     'decision_function_shape': ['ovo', 'ovr']
    # }
    # ]
    
    # SVMModel = GridSearchCV(SVC(), param_grid, cv=5)                          # performs grid search to find the best number of neighbors
    # SVMModel.fit(X_train, y_train)                                                   # fits the model to the training data
    # predictions = SVMModel.predict(X_test)                                           # makes predictions on the test data
    # print("\n\nBest SVM Parameters:", SVMModel.best_params_)                  # prints the best parameters found by the grid search
    # print("###########################Grid Search in " + file_path+ "###########################")
    # print("Best SVM Score From Grid Search:", SVMModel.best_score_ * 100, "%")# prints the best parameters found by the grid search
    # print(classification_report(y_test, predictions), "\n")                                 # prints the classification report of the model
    ######################################################################################

    
    ######################################################################################
    # SVM Model Training and Evaluation
    ######################################################################################
    # Initialize the SVM model
    #Parameters: based on grid search results
    
    if file_path == './results/umap_reduced_data.csv': #Best score: 96.92 from Grid Search
        print(BLUE + "###########################SVM in " + file_path+ "###########################" + RESET)
        print(BLUE + "SVM Parameters: {'C': 1, 'coef0': 3.0, 'degree': 4, 'gamma': 'auto', 'kernel': 'poly', 'tol': 0.0001}")
        svm_model = SVC(C = 1, coef0 = 3.0, degree = 4, gamma = 'auto', kernel = 'poly', tol = 0.0001)
        
    elif file_path == './results/tsne_reduced_data.csv':
        print(GREEN + "###########################SVM in " + file_path+ "###########################")
        print(GREEN + "SVM Parameters: {'C': 1, 'coef0': 3.0, 'gamma': 'auto', 'kernel': 'poly', 'tol': 0.0001}")
        svm_model = SVC(C = 1, coef0 = 3.0, gamma = 'auto', kernel = 'poly', tol = 0.0001)
        
    elif file_path == './results/pca_reduced_data.csv': #Best score: 94.74 from Grid Search
        print(BLUE + "###########################SVM in " + file_path+ "###########################" + RESET)
        print(BLUE + "SVM Parameters: {'C': 1, 'coef0': 2.0, 'degree': 4, 'gamma': '0.05', 'kernel': 'poly', 'tol': 0.0001}")
        svm_model = SVC(C = 1, coef0 = 2.0, gamma = 0.05, kernel = 'poly', tol = 0.0001)
        

    # Train the SVM model
    svm_model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = svm_model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy for {file_path}: {accuracy * 100:.2f}%\n\n' + RESET)
    ######################################################################################
    
    
