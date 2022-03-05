# Disaster Response Pipeline Project

### Summary :
This is a web application written in python using Flask to classify a text messages automatically into different categories (36) of disaster categories. There are two visualizations from the data set.
The project consists of 3 areas:
- ETL pipeline (process_data.py) to cleans data and store it.
- ML pipeline (train_classifier.py) to train and store a classifier model.
- Flask web app (run.py) to use a classification based on trained model.

### Packages Required
You need following python packages:
- flask
- plotly
- sqlalchemy
- pandas
- numpy
- sklearn
- nltk

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
