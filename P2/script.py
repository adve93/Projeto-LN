from nltk.tokenize import word_tokenize
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

# Load your training dataset and test dataset
train_data = pd.read_csv('C:/Users/afons/OneDrive/Documentos/GitHub/Projeto-LN/P2/train.txt', sep='\t', names=['label', 'review'])

# Read the test data, considering multi-line reviews separated by newlines
with open('C:/Users/afons/OneDrive/Documentos/GitHub/Projeto-LN/P2/test_just_reviews.txt', 'r') as file:
    lines = file.read().split('\n')
    lines.pop()

# Create a DataFrame for the test data
test_data = pd.DataFrame({'review': lines})


# Initialize the Porter Stemmer
stemmer = PorterStemmer()

# Initialize the WordNet Lemmatizer
lemmatizer = WordNetLemmatizer()

# Define a function for text preprocessing
def preprocess_text(text):

    # Lowe case the whole text
    text = text.lower()  

    # Define stop words
    stop_words = set(stopwords.words('english'))

    # Remove stop words
    words = text.split()
    words = [word for word in words if word not in stop_words]
    removed_stopwords = ' '.join(words)

    # Tokenize the text
    tokens = word_tokenize(removed_stopwords)

    # Apply stemming to the tokens
    stemmed_tokens = [stemmer.stem(token) for token in tokens]

    # Apply lemmatization to the tokens
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Join the tokens back into a string    
    processed_text = ' '.join(lemmatized_tokens)

    return text

# Apply text preprocessing to the 'review' column
train_data['review'] = train_data['review'].apply(preprocess_text)

# Apply the same text preprocessing to the test data
test_data['review'] = test_data['review'].apply(preprocess_text)  # Assuming preprocess_text is defined as in the previous code

# Vectorize the text data using TF-IDF
tfidf_vectorizer = TfidfVectorizer() 
X_train = tfidf_vectorizer.fit_transform(train_data['review'])
X_test = tfidf_vectorizer.transform(test_data['review'])

# Map labels to integers (e.g., TRUTHFULPOSITIVE: 0, TRUTHFULNEGATIVE: 1, DECEPTIVEPOSITIVE: 2, DECEPTIVENEGATIVE: 3)
label_mapping = {
    'TRUTHFULPOSITIVE': 0,
    'TRUTHFULNEGATIVE': 1,
    'DECEPTIVEPOSITIVE': 2,
    'DECEPTIVENEGATIVE': 3
}

label_mapping_reverse = {v: k for k, v in label_mapping.items()}

y_train = train_data['label'].map(label_mapping)

# Parameter grid to find best parameters
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': [0.1, 1, 10],
    'degree': [2, 3, 4],
    'coef0': [0.0, 1.0, 2.0]
}

#grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
#grid_search.fit(X_train, y_train)

#best_params = grid_search.best_params_

# Initialize an SVM classifier
svm_classifier = SVC(kernel='poly', C=0.1, coef0=2.0, degree=3, gamma=1)

# Initialize StratifiedKFold with 5 folds
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform k-fold cross-validation and get the accuracy scores for each label
misclassified_reviews = []

# Initialize an empty dictionary to store accuracy for each label
label_accuracies = {}

# Initialize an empty confusion matrix
confusion_matrix_total = None

# Initialize mean accuracy
mean_accuracy = 0

# Perform k-fold cross-validation and get the accuracy scores for each label
for train_index, test_index in stratified_kfold.split(X_train, y_train):
    X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
    y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]

    svm_classifier.fit(X_train_fold, y_train_fold)
    fold_predictions = svm_classifier.predict(X_test_fold)
    
    accuracy = accuracy_score(y_test_fold, fold_predictions)
    label_accuracies_fold = {}

    #Save missclassified reviews
    for i in range(len(y_test_fold)):
        true_label = y_test_fold.iloc[i]
        predicted_label = fold_predictions[i]
        if true_label != predicted_label:
            review_index = test_index[i]
            if 0 <= review_index < len(test_data):
                review = test_data.iloc[review_index]['review']
                misclassified_reviews.append((review, label_mapping_reverse[true_label], label_mapping_reverse[predicted_label]))
    
    for label, label_idx in label_mapping.items():
        label_mask = (y_test_fold == label_idx)
        label_accuracy = accuracy_score(y_test_fold[label_mask], fold_predictions[label_mask])
        label_accuracies_fold[label] = label_accuracy
    
    for label, accuracy in label_accuracies_fold.items():
        if label in label_accuracies:
            label_accuracies[label].append(accuracy)
        else:
            label_accuracies[label] = [accuracy]

    # Compute the confusion matrix for this fold
    confusion_matrix_fold = confusion_matrix(y_test_fold, fold_predictions)
    if confusion_matrix_total is None:
        confusion_matrix_total = confusion_matrix_fold
    else:
        confusion_matrix_total += confusion_matrix_fold



# Print the accuracy for each label
for label, accuracies in label_accuracies.items():
    mean_label_accuracy = sum(accuracies) / len(accuracies)
    mean_accuracy = mean_accuracy + mean_label_accuracy
    print(f'Accuracy for {label}: {mean_label_accuracy:.2f}')

# Calculate the mean accuracy across all folds
mean_accuracy = mean_accuracy/4
print(f'Overall accuracy: {mean_accuracy:.2f}')

# Abbreviated labels
abbreviated_labels = ['TP', 'TN', 'DP', 'DN']

# Create a DataFrame for the confusion matrix with labeled rows and columns
confusion_matrix_df = pd.DataFrame(confusion_matrix_total, columns=abbreviated_labels, index=abbreviated_labels)

# Print the confusion matrix with labels
print("Confusion Matrix:")
print(confusion_matrix_df)

# Write the misclassified reviews to the file
with open('misclassified_reviews.txt', 'w', encoding='utf-8') as file:
    for review, true_label, predicted_label in misclassified_reviews:
        file.write(f'Review: {review}\n')
        file.write(f'True Label: {true_label}\n')
        file.write(f'Predicted Label: {predicted_label}\n\n')

print(len(misclassified_reviews))

# Fit SVM model
svm_classifier.fit(X_train, y_train)

# Make predictions on the test dataset and validation set
test_predictions = svm_classifier.predict(X_test)

# Map integer predictions back to labels
test_data['predicted_label'] = [label_mapping_reverse[label] for label in test_predictions]

# Save the results or print them
test_data.to_csv('test_results.csv', index=False)