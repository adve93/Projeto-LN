import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load your training dataset and test dataset
train_data = pd.read_csv('train.txt', sep='\t', names=['label', 'review'])
# Read the test data while considering multiline reviews
with open('test_just_reviews.txt', 'r') as file:
    lines = file.read().split('\n')
    test_data = {'review': []}
    current_review = ""
    for line in lines:
        if line.strip():  # Check if the line is not empty
            current_review += line.strip() + ' '
        else:
            test_data['review'].append(current_review.strip())
            current_review = ""

# Create a DataFrame for the test data
test_data = pd.DataFrame(test_data)

# Preprocess the text data (tokenization, lowercasing, etc.)
# You may need to define a function for this based on your specific requirements.

# Vectorize the text data using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # You can adjust the number of features
X_train = tfidf_vectorizer.fit_transform(train_data['review'])
X_test = tfidf_vectorizer.transform(test_data['review'])

# Map labels to binary values (TRUTHFUL as 1 and DECEPTIVE as 0)
train_data['label'] = train_data['label'].apply(lambda x: 1 if 'TRUTHFUL' in x else 0)

# Train an SVM classifier
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, train_data['label'])

# Make predictions on the test dataset
test_predictions = svm_classifier.predict(X_test)

# Map binary predictions back to labels
test_data['predicted_label'] = ['TRUTHFUL' if label == 1 else 'DECEPTIVE' for label in test_predictions]

# Save the results or print them
test_data.to_csv('test_results.csv', index=False)

# You can also evaluate the model's performance using additional metrics.
# For binary classification, accuracy is a common metric.
# true_labels = ...  # Ground truth labels for the test dataset
# accuracy = accuracy_score(true_labels, test_predictions)
# print("Accuracy:", accuracy)
