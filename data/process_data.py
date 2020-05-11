# import libraries
import sys

import pandas as pd
from sqlalchemy import create_engine


def load_data(msg_path, cat_path):
    # load messages dataset
    messages = pd.read_csv(msg_path)

    # load categories dataset
    categories = pd.read_csv(cat_path)

    # merge datasets
    # Dropped the origial as it is in another language
    df = messages.drop(columns=['original'], axis=1).merge(categories, on='id')

    return df


def get_colname_colval(cat):
    '''
    Function to split the categories to columns
        cat : input - Categories Series
    Output:
        colname : Name of the column
        colval : Values of the corresponding column
    '''
    colname = []
    colval = []
    # Get the column Name
    for col in cat[0]:
        colname.append(col.split('-')[0])

    # Create a list of values for the column
    for col in cat:
        v = []
        for val in col:
            v.append(int(val.split('-')[1]))
        colval.append(v)

    categories = pd.DataFrame(colval, columns=colname)
    return categories


def clean_data(df):
    # create a dataframe of the 36 individual category columns
    categories = get_colname_colval(df['categories'].str.split(';'))
    # drop the original categories column from `df`
    df.drop(['categories'], axis=1, inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # check number of duplicates
    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, db_path):
    engine = create_engine('sqlite:///' + db_path)
    df.to_sql('Disaster_DB', engine, index=False)
    pass


def main():
    if len(sys.argv) == 4:

        msg_path, cat_path, db_path = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(msg_path, cat_path))
        df = load_data(msg_path, cat_path)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(db_path))
        save_data(df, db_path)

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