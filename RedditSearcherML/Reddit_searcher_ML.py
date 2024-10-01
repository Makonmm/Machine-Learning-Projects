import re
import praw
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

infos = ['datascience', 'machinelearning',
         'astronomy', 'physics', 'conspiracy']


def load_data():
    reddit = praw.Reddit(
        client_id=" ",
        client_secret=" ",
        password=" ",
        user_agent="USERAGENT",
        username=" ",
    )

    def char_count(post):
        return len(re.sub(r'\W\d', '', post.selftext))

    def mask(post):
        return char_count(post) >= 100

    data = []
    labels = []

    for i, info in enumerate(infos):
        subreddit_data = reddit.subreddit(info).new(limit=1000)
        posts = [post.selftext for post in filter(mask, subreddit_data)]

        data.extend(posts)
        labels.extend([i] * len(posts))

        print(f"Number of posts r/{info}: {len(posts)}")
        print(f"\nThere is a post extracted: {posts[0][:600]}...\n")
        print("_" * 80 + '\n')

    return data, labels


TEST_SIZE = 0.2
RANDOM_STATE = 0


def split_data(data, labels):
    print(f"Splitting {
          100 * TEST_SIZE}% of the data for test and evaluation")

    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"Test amount: {len(y_test)}")

    return X_train, X_test, y_train, y_test


MIN_DOC_FREQ = 1
N_COMPONENTS = 1000
N_ITER = 30


def preprocessing_pipeline():
    pattern = r'\W\d|http.*|www.*\s+'

    def preprocessor(text):
        return re.sub(pattern, '', text)

    vectorizer = TfidfVectorizer(
        preprocessor=preprocessor, stop_words='english', min_df=MIN_DOC_FREQ
    )
    decomposition = TruncatedSVD(n_components=N_COMPONENTS, n_iter=N_ITER)

    pipeline = Pipeline([('tfidf', vectorizer), ('svd', decomposition)])

    return pipeline


N_NEIGHBORS = 4
CV = 3


def create_models():
    model_1 = KNeighborsClassifier(n_neighbors=N_NEIGHBORS)
    model_2 = RandomForestClassifier(random_state=RANDOM_STATE)
    model_3 = LogisticRegressionCV(cv=CV, random_state=RANDOM_STATE)

    models = [("KNN", model_1), ("RandomForest", model_2), ("LogReg", model_3)]

    return models


def train_evaluate(models, pipeline, X_train, X_test, y_train, y_test):

    results = []

    for name, model in models:

        steps = pipeline.steps + [(name, model)]
        pipe = Pipeline(steps)

        print(f"Training the model {name} ")
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)

        report = classification_report(y_test, y_pred)
        print(f"Classification report\n: {report}")

        results.append(
            [model, {'model': name, 'predictions': y_pred, 'report': report, }])
    return results


if __name__ == "__main__":

    data, labels = load_data()
    X_train, X_test, y_train, y_test = split_data(data, labels)

    pipeline = preprocessing_pipeline()

    all_models = create_models()

    results = train_evaluate(all_models, pipeline,
                             X_train, X_test, y_train, y_test)

print("Success")
