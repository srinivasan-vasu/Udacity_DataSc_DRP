import sys

import nltk

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from sqlalchemy import create_engine
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re, pickle

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import precision_recall_fscore_support

# from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.tree import DecisionTreeClassifier

url_regex = 'http[s]?://(?:[A-Za-z]|[0-9]|[$-._@+&]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


def load_data(db_path):
    # load data from database
    engine = create_engine('sqlite:///' + db_path)
    df = pd.read_sql_table('Disaster_DB', con=engine)

    X = df['message']
    y = df[df.columns[4:]]
    cat_names = y.columns
    return X, y, df.columns


def tokenize(text):
    url = re.findall(url_regex, text)
    for u in url:
        text = text.replace(u, 'urlplaceholder')

    token = word_tokenize(text)
    lemmatize = WordNetLemmatizer()

    clean_token = []
    for tok in token:
        clean_token.append(lemmatize.lemmatize(tok).lower().strip())

    return clean_token


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {
        # 'vect__ngram_range': ((1, 1),(1,2)),
        # 'vect__max_df' : (0.5, 0.75, 1.0),
        # 'tfidf__use_idf': (True, False),
        # 'vect__max_features': (None, 5000, 10000),
        'clf__estimator__n_estimators': [50, 100, 200],
        'clf__estimator__min_samples_split': [2, 3, 4]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def display_results(y_test, y_pred):
    id = 0
    results = pd.DataFrame(columns=['Category', 'Precision', 'Recall', 'F1-Score'])
    for col in y_test:
        precision, recall, f1, support = precision_recall_fscore_support(y_test[col], y_pred[:, id], average='weighted')
        results.at[id + 1, 'Category'] = col
        results.at[id + 1, 'Precision'] = precision
        results.at[id + 1, 'Recall'] = recall
        results.at[id + 1, 'F1-Score'] = f1
        id = id + 1
    print('Average Precision: ', results['Precision'].mean())
    print('Average Recall: ', results['Recall'].mean())
    print('Average f1: ', results['F1-Score'].mean())

    return results


def evaluate_model(model, X_test, y_test, category_names):
    y_pred = model.predict(X_test.values)
    results = display_results(y_test, y_pred)
    pass


def save_model(model, mdl_path):
    pickle.dump(model, open(mdl_path, 'wb'))


def main():
    if len(sys.argv) == 3:
        db_path, mdl_path = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(db_path))
        X, y, category_names = load_data(db_path)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(mdl_path))
        save_model(model, mdl_path)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
