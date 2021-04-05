# Project: Disaster Response Pipeline
# Project overview
I found a data set containing real messages that were sent during disaster events. I created a machine learning pipeline to categorize these events so that I can send the messages to an appropriate disaster relief agency. in include a web app where an emergency worker can input a new message and get classification results in several categories. The web app display visualizations of the data.
# Installation
The code should run using Python versions 3.*. The necessary libraries are:
sklearn
sys
re
pandas
numpy
spite3
nilk
sqlalchemy 
pickle
json
plotly
flask

# Quick Start
Run the following commands in the project's root directory to set up database and model.

```
python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
python train_classifier.py ../data/DisasterResponse.db classifier.pkl
python run.py
```
## 1. ETL Pipeline
In a Python script, process_data.py, write a data cleaning pipeline that:
Loads the messages and categories datasets
Merges the two datasets
Cleans the data
Stores it in a SQLite database

## 2. ML Pipeline

In a Python script, train_classifier.py, write a machine learning pipeline that:
Loads data from the SQLite database
Splits the dataset into training and test sets
Builds a text processing and machine learning pipeline
Trains and tunes a model using GridSearchCV
Outputs results on the test set
Exports the final model as a pickle file

## 3. Flask Web App
 data visualizations using Plotly in the web app
 
# files

Here's the file structure of the project:

- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- DisasterResponse.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model 

- README.md


# Result:




# Licensing, Acknowledgements
Must give credit to figure-8 that provided the data used in this repository. Feel free to use the code in this repository as you would like!


