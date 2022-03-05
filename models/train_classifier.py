import sys
import pandas as pd
import re
import pickle

from sqlalchemy import create_engine
from sqlalchemy import inspect
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


def load_data(database_filepath):
    '''
    load data from sql database,
    and return feature dataframe, label-data DataFrame, labels as list
    
    Parameters
    ----------
    database_filename : str
        The path of the database
    '''

    engine = create_engine('sqlite:///' + database_filepath)
    inspector = inspect(engine)
    df = pd.read_sql_table(inspector.get_table_names()[0], con = engine)
    X = df.message.values
    Y = df.iloc[:,4:]
    return X, Y, Y.columns

def tokenize(text):
    '''
    tokenize text
    
    Parameters
    ----------
    text : str
        The text to be tokenized
    '''

    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        
        if tok in stopwords.words("english"):
            continue
            
        tok = PorterStemmer().stem(tok)
        
        # Reduce words to their root form
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        
        clean_tokens.append(clean_tok)
        
    clean_tokens = [tok for tok in clean_tokens if tok.isalpha()]
    return clean_tokens



def build_model():
    '''
    build pipeline, set parameter, do Gridsearch and return the model.
    '''

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf',  MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'vect__max_df':[0.80,1.0],
        'clf__estimator__n_estimators': [30, 50]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=10, n_jobs=4)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    evaluate model by sklearn classification_report
    
    Parameters
    ----------
    model : model
        The model 
    X_test : array
        The test data set.
    Y_test : dataframe
        The set of labels to all the data in x_test.
    category_names : list
        The list of the categories
    '''

    Y_pred = model.predict(X_test)
    for ix, col in enumerate(category_names):
        print(col)
        print(classification_report(Y_test[col], Y_pred[:,ix]))
    print("Accuracy: ",(Y_pred == Y_test).mean().mean())


def save_model(model, model_filepath):
    '''
    save the model as pickle file
    
    Parameters
    ----------
    model : model
        The model
    X_test : str
        The is the test data set.
    '''
    
    pickle.dump(model, open(model_filepath, 'wb'))


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