import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical


# Step 3: Load the Wine dataset
wine = load_wine()

X = wine.data
y = wine.target


# Step 4: Display dataset information
print("Dataset Information")
print("-------------------")
print("Number of samples:", X.shape[0])
print("Number of features:", X.shape[1])
print("Number of classes:", len(np.unique(y)))

print("\nFeature Names:")
print(wine.feature_names)

print("\nTarget Class Names:")
print(wine.target_names)


# Step 5: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# Step 6: Standardize the features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Step 7: One-hot encode target classes
y_train_encoded = to_categorical(y_train, num_classes=3)
y_test_encoded = to_categorical(y_test, num_classes=3)


# Step 8: Create the Feedforward Neural Network
model = Sequential([
    Dense(16, activation='relu', input_shape=(13,)),
    Dense(8, activation='relu'),
    Dense(3, activation='softmax')
])


# Step 9: Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nModel Architecture:")
model.summary()


# Step 10: Train the model
history = model.fit(
    X_train,
    y_train_encoded,
    epochs=100,
    batch_size=16,
    validation_split=0.20,
    verbose=1
)


# Step 11: Evaluate the model
test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test_encoded,
    verbose=0
)

print("\nTest Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
print("Test Accuracy (%):", test_accuracy * 100)


# Step 12: Make predictions
y_pred_prob = model.predict(X_test, verbose=0)

y_pred = np.argmax(y_pred_prob, axis=1)

print("\nPredicted Classes:")
print(y_pred)

print("\nActual Classes:")
print(y_test)


# Step 13: Calculate classification accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nClassification Accuracy:", accuracy)
print("Classification Accuracy (%):", accuracy * 100)


# Step 14: Classification report
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=wine.target_names
    )
)


# Step 15: Confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=wine.target_names
)

disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()


# Step 16: Plot training and validation accuracy
plt.figure(figsize=(8, 5))

plt.plot(
    history.history['accuracy'],
    label='Training Accuracy'
)

plt.plot(
    history.history['val_accuracy'],
    label='Validation Accuracy'
)

plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()


# Step 17: Plot training and validation loss
plt.figure(figsize=(8, 5))

plt.plot(
    history.history['loss'],
    label='Training Loss'
)

plt.plot(
    history.history['val_loss'],
    label='Validation Loss'
)

plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()


# Step 18: Record the final result
print("\nFinal Result")
print("------------")
print(f"Test Loss      : {test_loss:.4f}")
print(f"Test Accuracy  : {test_accuracy:.4f}")
print(f"Accuracy       : {test_accuracy * 100:.2f}%")
