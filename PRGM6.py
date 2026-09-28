from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np

# 1. Load the Diabetes dataset
diabetes = load_diabetes()

# 2. Separate features and target
X = diabetes.data
y = diabetes.target

# Convert the continuous target into two classes
# 0 = Low, 1 = High
y = (y > np.median(y)).astype(int)

# 3 & 4. Split the dataset
# 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Create the Gaussian Naive Bayes classifier
gnb = GaussianNB()

# 6. Train the model
gnb.fit(X_train, y_train)

# 7. Predict the classes of test data
y_pred = gnb.predict(X_test)

# 8. Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Display actual and predicted labels
print("Actual Labels:")
print(y_test)

print("\nPredicted Labels:")
print(y_pred)

# Display accuracy
print("\nAccuracy of Gaussian Naive Bayes Classifier: {:.2f}%".format(
    accuracy * 100
))

# 9. Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 10. Classification Report
print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Low", "High"]
))