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

train_data = pd.read_csv('C:/Users/afons/OneDrive/Documentos/GitHub/Projeto-LN/P2/train.txt', sep='\t', names=['label', 'review'])

with open('C:/Users/afons/OneDrive/Documentos/GitHub/Projeto-LN/P2/test_just_reviews.txt', 'r') as file:
    lines = file.read().split('\n')
    lines.pop()

test_data = pd.DataFrame({'review': lines})

stemmer = PorterStemmer()

lemmatizer = WordNetLemmatizer()

def preprocess_text(text):

    text = text.lower()  

    stop_words = set(stopwords.words('english'))

    words = text.split()
    words = [word for word in words if word not in stop_words]
    removed_stopwords = ' '.join(words)

    tokens = word_tokenize(removed_stopwords)

    stemmed_tokens = [stemmer.stem(token) for token in tokens]

    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

    processed_text = ' '.join(lemmatized_tokens)

    return text

train_data['review'] = train_data['review'].apply(preprocess_text)

test_data['review'] = test_data['review'].apply(preprocess_text) 

tfidf_vectorizer = TfidfVectorizer() 
X_train = tfidf_vectorizer.fit_transform(train_data['review'])
X_test = tfidf_vectorizer.transform(test_data['review'])

label_mapping = {
    'TRUTHFULPOSITIVE': 0,
    'TRUTHFULNEGATIVE': 1,
    'DECEPTIVEPOSITIVE': 2,
    'DECEPTIVENEGATIVE': 3
}

label_mapping_reverse = {v: k for k, v in label_mapping.items()}

y_train = train_data['label'].map(label_mapping)

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

svm_classifier = SVC(kernel='poly', C=0.1, coef0=2.0, degree=3, gamma=1)

stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

misclassified_reviews = []

label_accuracies = {}

confusion_matrix_total = None

mean_accuracy = 0

for train_index, test_index in stratified_kfold.split(X_train, y_train):
    X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
    y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]

    svm_classifier.fit(X_train_fold, y_train_fold)
    fold_predictions = svm_classifier.predict(X_test_fold)
    
    accuracy = accuracy_score(y_test_fold, fold_predictions)
    label_accuracies_fold = {}

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

    confusion_matrix_fold = confusion_matrix(y_test_fold, fold_predictions)
    if confusion_matrix_total is None:
        confusion_matrix_total = confusion_matrix_fold
    else:
        confusion_matrix_total += confusion_matrix_fold

for label, accuracies in label_accuracies.items():
    mean_label_accuracy = sum(accuracies) / len(accuracies)
    mean_accuracy = mean_accuracy + mean_label_accuracy
    print(f'Accuracy for {label}: {mean_label_accuracy:.2f}')

mean_accuracy = mean_accuracy/4
print(f'Overall accuracy: {mean_accuracy:.2f}')

abbreviated_labels = ['TP', 'TN', 'DP', 'DN']

confusion_matrix_df = pd.DataFrame(confusion_matrix_total, columns=abbreviated_labels, index=abbreviated_labels)

print("Confusion Matrix:")
print(confusion_matrix_df)

with open('misclassified_reviews.txt', 'w', encoding='utf-8') as file:
    for review, true_label, predicted_label in misclassified_reviews:
        file.write(f'Review: {review}\n')
        file.write(f'True Label: {true_label}\n')
        file.write(f'Predicted Label: {predicted_label}\n\n')

print(len(misclassified_reviews))

svm_classifier.fit(X_train, y_train)

test_predictions = svm_classifier.predict(X_test)

result = [label_mapping_reverse[label] for label in test_predictions]

with open("result.txt", "w") as file:
    for item in result:
        file.write(item + "\n")

file.close()

