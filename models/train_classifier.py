import sys
import nltk
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle
import sqlite3


nltk.download(['punkt', 'wordnet', 'stopwords'])


def load_data(database_filepath):
    '''
    INPUT: the data location
    
    OUTPUT: the dataframe of message, the dataframe of categores, categores' name
 
    Description:Load data from database and separate it into X, Y.
    
    '''
    
    # Creating the Connection object of sqlite Database
    conn = sqlite3.connect(database_filepath)
    # getting data from sqlite data base
    df = pd.read_sql('SELECT * FROM Disaster',conn)
    # Define feature and target variables X and Y
    X = df['message']
    Y = df.drop(columns = ['id', 'message', 'original', 'genre' ])
    columns = list(set(df.columns) - set(['id', 'message', 'original', 'genre' ]))
    return X, Y ,columns

def tokenize(text):
    
    '''
    INPUT: a message
    
    OUTPUT: tokenize text
 
    Description:NLP
    
    '''
    
    
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize andremove stop words
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens

def build_model():
    
    '''
    
    OUTPUT: model
 
    Description:build model using pipeline and GridSearchCV
    
    '''
    
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(estimator=RandomForestClassifier(n_jobs = 1),n_jobs = 1))])
    
    parameters = {
    'vect__max_df': (0.5, 1.0),
    'tfidf__use_idf': (True, False),
    'clf__estimator__n_estimators': [5,10]}

    model = GridSearchCV(estimator = pipeline, param_grid=parameters,n_jobs=1)
    
    
    return model

def evaluate_model(model, X_test, Y_test, category_names):
    
    '''
    Description:evaluate model using classification_report, Report the f1 score, precision and recall for each output category of the dataset.
    '''
    
    Y_pred = model.predict(X_test)
    for i in range(len(category_names)):
        classification_report(Y_test[[category_names[i]]], Y_pred[:, i])

def save_model(model, model_filepath):
    '''
    Description: Export your model as a pickle file
    '''
    list_pickle = open(model_filepath, 'wb')
    pickle.dump(model, list_pickle)
    list_pickle.close()

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()