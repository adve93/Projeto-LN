from nltk.tokenize import word_tokenize
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from sklearn.model_selection import GridSearchCV

# Load your training dataset and test dataset
train_data = pd.read_csv('train.txt', sep='\t', names=['label', 'review'])

# Read the test data, considering multi-line reviews separated by newlines
with open('test_just_reviews.txt', 'r') as file:
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
    processed_text = ' '.join(stemmed_tokens)

    return text

# Apply text preprocessing to the 'review' column
train_data['review'] = train_data['review'].apply(preprocess_text)

# Apply the same text preprocessing to the test data
test_data['review'] = test_data['review'].apply(preprocess_text)  # Assuming preprocess_text is defined as in the previous code

# Vectorize the text data using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000) 
X_train = tfidf_vectorizer.fit_transform(train_data['review'])
X_test = tfidf_vectorizer.transform(test_data['review'])

# Map labels to integers (e.g., TRUTHFULPOSITIVE: 0, TRUTHFULNEGATIVE: 1, DECEPTIVEPOSITIVE: 2, DECEPTIVENEGATIVE: 3)
label_mapping = {
    'TRUTHFULPOSITIVE': 0,
    'TRUTHFULNEGATIVE': 1,
    'DECEPTIVEPOSITIVE': 2,
    'DECEPTIVENEGATIVE': 3
}
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

#print(best_params)

# Initialize an SVM classifier
svm_classifier = SVC(kernel='poly', C=0.1, coef0=2.0, degree=3, gamma=1)

# Initialize StratifiedKFold with 5 folds
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform k-fold cross-validation and get the accuracy scores
accuracy_scores = cross_val_score(svm_classifier, X_train, y_train, cv=stratified_kfold, scoring='accuracy')

# Calculate the mean accuracy across all folds
mean_accuracy = accuracy_scores.mean()
print(f'Mean Accuracy: {mean_accuracy:.2f}')

# Fit SVM model
svm_classifier.fit(X_train, y_train)

# Make predictions on the test dataset and validation set
test_predictions = svm_classifier.predict(X_test)

# Map integer predictions back to labels
label_mapping_reverse = {v: k for k, v in label_mapping.items()}
test_data['predicted_label'] = [label_mapping_reverse[label] for label in test_predictions]

# Save the results or print them
test_data.to_csv('test_results.csv', index=False)