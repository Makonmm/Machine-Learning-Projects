# Version 1 (Voting)

import numpy as np
from nltk.corpus import stopwords
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Loading data

news = load_files('data', encoding='utf-8', decode_error='replace')

X = news.data
y = news.target

# Splitting test and training data

my_stop_words = set(stopwords.words('english'))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=75)

# Vectorizing

vectorizer = TfidfVectorizer(
    norm=None, stop_words=my_stop_words, max_features=1000, decode_error='ignore')

X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# Training

model_1 = LogisticRegression(
    multi_class='multinomial', solver='lbfgs', random_state=30, max_iter=1000)
model_2 = RandomForestClassifier(
    n_estimators=1000, max_depth=100, random_state=1)
model_3 = MultinomialNB()

result = []

voting_model = VotingClassifier(
    estimators=[('lg', model_1), ('rf', model_2), ('nb', model_3)], voting='soft')
print("\n Voting model \n")
print(voting_model)

voting_model = voting_model.fit(X_train_vectors, y_train)
previsions = voting_model.predict(X_test_vectors)

result.append(accuracy_score(y_test, previsions))

print("\n Model accuracy:", accuracy_score(y_test, previsions))
