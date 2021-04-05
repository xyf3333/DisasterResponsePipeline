import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from plotly.graph_objs import Pie

#from sklearn.externals 
import joblib
from sqlalchemy import create_engine
import pickle


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Disaster', engine)


# load model
model = joblib.load("../models/classifier.pkl").set_params(n_jobs=1)
#model = pickle.load(open('../models/classifier.pkl','rb'))


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals

    request_counts = df.groupby('request').count()['message']
    request_names = ['yes' if int(i) == 1 else 'No' for i in list(request_counts.index)]

    food_counts = df.groupby('food').count()['message']
    food_names = ['yes' if int(i) == 1 else 'No' for i in list(food_counts.index)]
    
    water_counts = df.groupby('water').count()['message']
    water_names = ['yes' if int(i) == 1 else 'No' for i in list(water_counts.index)]
    
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Pie(
                    labels = request_names,
                    values = request_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message request',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Request"
                }
            }
        },
        {
             'data':[
                Bar(
                    x=food_names,
                    y=food_counts, 
                    name = 'Food'
                ),
                 Bar(
                    x=water_names,
                    y=water_counts,
                    name = 'Water'
                )
     
             ],
            'layout': {
                'title': 'Distribution of food and water',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': " "
                }
            }
     
        }
        
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()