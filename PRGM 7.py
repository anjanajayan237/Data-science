# Text Classification using SVM
# Dataset: 20 Newsgroups Dataset

# 1. Import required libraries
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report


# 2. Select the news categories
categories = [
    'comp.graphics',
    'rec.sport.baseball',
    'sci.space'
]


# 3. Load the training dataset
train_data = fetch_20newsgroups(
    subset='train',
    categories=categories,
    remove=('headers', 'footers', 'quotes')
)


# 4. Load the testing dataset
test_data = fetch_20newsgroups(
    subset='test',
    categories=categories,
    remove=('headers', 'footers', 'quotes')
)


# 5. Display number of documents
print("Number of training documents:", len(train_data.data))
print("Number of testing documents:", len(test_data.data))


# 6. Create TF-IDF Vectorizer
vectorizer = TfidfVectorizer(
    stop_words='english'
)
# 7. fit TF-IDF ON training documents
X_train = vectorizer.fit_transform(train_data.data)


# 8. Transform test documents
X_test = vectorizer.transform(test_data.data)


# 9. Get target labels
y_train = train_data.target
y_test = test_data.target


# 10. Create Linear SVM classifier
svm_model = LinearSVC()


# 11. Train the SVM model
svm_model.fit(X_train, y_train)


# 12. Predict the test documents
y_pred = svm_model.predict(X_test)


# 13 & 14. Calculate classification accuracy
accuracy = accuracy_score(y_test, y_pred)


# 15. Generate classification report
report = classification_report(
    y_test,
    y_pred,
    target_names=train_data.target_names
)


# 16. Display results
print("\n--- SVM Text Classification Results ---")

print("\nCategories:")
for category in train_data.target_names:
    print("-", category)

print("\nTraining documents:", len(train_data.data))
print("Testing documents:", len(test_data.data))

print("\nAccuracy: {:.2f}%".format(accuracy * 100))

print("\nClassification Report:")