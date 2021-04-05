import sys
import re
import sqlite3
import pandas as pd
import numpy as np
import sqlite3

def load_data(messages_filepath, categories_filepath):
    '''
    INPUT:
    messages_filepath - the csv file provide messages data.
    categories_filepath - the csv file provide categories data.
    
    OUTPUT:
    
    
    Description:
    Provides a list of the article_ids and article titles that have been seen by a user
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how ='left', on = 'id')
    return df
def clean_data(df):
    categories = df.categories.str.split(';', expand = True)
    row = categories.iloc[0]
    category_colnames = [re.sub('-.', '', x) for x in row]
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    # set each value to be the last character of the string 
    # convert column from string to numeric
    for column in categories:
        list_x = []
        for x in categories[column]:
            list_x.append(x[-1])
        categories[column] = list_x
        
    df.drop(columns = ['categories'], inplace=True)
    df = pd.concat([df,categories], axis = 1)
    
    # drop duplicates
    df.drop_duplicates( inplace=True)
    
    return df

def save_data(df, database_filename):

    # Creating sqlite DataBase Connection Object
    conn = sqlite3.connect(database_filename)
    
    # Dumping all cleaned data to sqlite database
    df.to_sql('Disaster', conn, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()