import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer

# Load the cancer data
data = load_breast_cancer()
target = pd.Series(data.target)
data = pd.DataFrame(data.data, columns=data.feature_names)
data.index = data.index.astype(str)

# Scale the data using MinMaxScaler
X = data
y = target
# Perceptrons
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=35)

# Initialize the Perceptron model
perceptron = Perceptron()
# Train the Perceptron model
perceptron.fit(X_train, y_train)
# Make predictions on the test set
y_pred = perceptron.predict(X_test)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

