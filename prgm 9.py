# Decision Tree Classification using Iris Dataset

# Step 1: Import required libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt


# Step 2: Load the Iris dataset
iris = load_iris()

# Separate input features (X) and target classes (y)
X = iris.data
y = iris.target


# Step 3: Display dataset information
print("Number of samples:", X.shape[0])
print("Number of features:", X.shape[1])
print("Feature names:", iris.feature_names)
print("Class names:", iris.target_names)


# Function to train and evaluate the Decision Tree
def decision_tree_experiment(test_size):
    print("\n" + "=" * 60)
    print(f"Train-Test Split: {int((1-test_size)*100)}% Training / "
          f"{int(test_size*100)}% Testing")
    print("=" * 60)

    # Step 4: Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # Step 5: Create Decision Tree classifier using Gini criterion
    model = DecisionTreeClassifier(
        criterion="gini",
        random_state=42
    )

    # Step 6: Train the model
    model.fit(X_train, y_train)

    # Step 7: Predict classes of test data
    y_pred = model.predict(X_test)

    # Step 8 & 9: Compare predictions and calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print("\nActual classes:   ", y_test)
    print("Predicted classes:", y_pred)

    print("\nClassification Accuracy:",
          f"{accuracy * 100:.2f}%")

    # Step 10: Display confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    print("\nConfusion Matrix:")
    print(cm)

    # Step 11: Display classification report
    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names
    ))

    # Step 12 & 13: Visualize the Decision Tree
    plt.figure(figsize=(15, 8))

    plot_tree(
        model,
        feature_names=iris.feature_names,
        class_names=iris.target_names,
        filled=True,
        rounded=True,
        fontsize=9
    )

    plt.title(
        f"Decision Tree - {int((1-test_size)*100)}:{int(test_size*100)} Train-Test Split"
    )
    plt.show()

    return accuracy


# Run experiment with 80% training and 20% testing
accuracy_80_20 = decision_tree_experiment(0.20)


# Run experiment with 70% training and 30% testing
accuracy_70_30 = decision_tree_experiment(0.30)


# Step 14: Display final results
print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)

print(f"80% Training / 20% Testing Accuracy: "
      f"{accuracy_80_20 * 100:.2f}%")

print(f"70% Training / 30% Testing Accuracy: "
      f"{accuracy_70_30 * 100:.2f}%")

