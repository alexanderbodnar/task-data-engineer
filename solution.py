from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sqlalchemy import create_engine

import pandas as pd
import traceback
import logging
import joblib
import sys

MODEL_NAME              = "model.joblib"
DATASET                 = "housing.csv"
DATASET_PREPROCESSED    = "housing_preprocessed.csv"
PREDICTIONS             = "housing_predictions.csv"
RANDOM_STATE            = 100

# Database config
TABLE_NAME_PREPROCESSED = "HOUSING_DATA_PREPROCESSED"
TABLE_NAME_PREDICTION   = "HOUSING_DATA_PREDICTED"

DATABASE_URL = 'postgresql://postgres:alex@localhost:5432/data_engineer'

# Logging config
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# ---------------------------------------------------------------------------------------------------
def preprocess_housing(filepath: str):
    
    df = pd.read_csv(filepath)
    
    # AGENCY column sa medzi fitoch nenachadza v modeli
    df = df.drop(columns=['AGENCY'])

    print("NULL values remaining:\n", df.isnull().sum()[df.isnull().sum() > 0])

    # Osetrenie NaN hodnot a dopocitanie prostrednych hodnot
    df.fillna(df.median(numeric_only=True), inplace=True)

    # V datasete sa nachadzaju 'Null' stringy. Nahradenie za numeric NaN
    df.replace('Null', pd.NA, inplace=True)

    # Konverzia vsetkych columns okrem ocean_proximity na numeric
    df[df.columns[:-1]] = df[df.columns[:-1]].apply(pd.to_numeric, errors='coerce')

    # Dopocitanie prostrednych hodnot pre zvysne NaN hodnoty
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Odstranenie 1 NaN riadku
    df.dropna(subset=['OCEAN_PROXIMITY'], inplace=True)

    # Print remaining NULL values (if any)
    print("NULL values remaining:\n", df.isnull().sum()[df.isnull().sum() > 0])


    # Premenovanie columns kvoli column names uz v nafitovanom modeli
    df = df.rename(columns={
        'LONGITUDE': 'longitude',
        'LAT': 'latitude',
        'MEDIAN_AGE': 'housing_median_age',
        'ROOMS': 'total_rooms',
        'BEDROOMS': 'total_bedrooms',
        'POP': 'population',
        'HOUSEHOLDS': 'households',
        'MEDIAN_INCOME': 'median_income',
        'OCEAN_PROXIMITY': 'ocean_proximity',
        'MEDIAN_HOUSE_VALUE': 'median_house_value',
    })
    print(df.columns)

    # Kategoricky label stlpec encodeneme na 0,1
    df = pd.get_dummies(df, columns=['ocean_proximity'], prefix='ocean_proximity')
    expected_ocean_columns = [
        'ocean_proximity_<1H OCEAN',
        'ocean_proximity_INLAND',
        'ocean_proximity_ISLAND',
        'ocean_proximity_NEAR BAY',
        'ocean_proximity_NEAR OCEAN'
    ]

    for col in expected_ocean_columns:
        if col not in df.columns:
            df[col] = 0


    # Finalny df a zoradene stlpce X + Y (posledny stlpec)
    preprocessed_df = df[['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income'] + expected_ocean_columns + ['median_house_value']]
    print(preprocessed_df.columns)

    preprocessed_df.to_csv("housing_preprocessed.csv", index=False)
    return preprocessed_df

def prepare_data(input_data_path):
    df=pd.read_csv(input_data_path)
    df=df.dropna() 

    # encode the categorical variables
    df = pd.get_dummies(df)

    df_features=df.drop(['median_house_value'],axis=1)
    y=df['median_house_value'].values

    X_train, X_test, y_train, y_test = train_test_split(df_features, y, test_size=0.2, random_state=RANDOM_STATE)

    return (X_train, X_test, y_train, y_test)

def train(X_train, y_train):
    # what columns are expected by the model
    X_train.columns

    regr = RandomForestRegressor(max_depth=12)
    regr.fit(X_train,y_train)

    return regr

def predict(X, model):
    Y = model.predict(X)
    return Y

def save_model(model, filename):
    with open(filename, 'wb'):
        joblib.dump(model, filename, compress=3)

def load_model(filename):
    model = joblib.load(filename)
    return model
# ---------------------------------------------------------------------------------------------------

if __name__ == "__main__":  
    try:
        logging.info("Preprocessing the raw dataset")
        # Preprocessing a transformacia datasetu
        preprocessed_df = preprocess_housing(filepath=DATASET)
        
        # Ulozenie transformnuteho datasetu do db.
        # (Ja som pouzil PostgreSQL lokalne u seba. Aby toto fungovalo je nutne si to nainstalovat a nakonfigurovat)
        logging.info("Saving the transformed dataset to db")
        engine = create_engine(DATABASE_URL)
        preprocessed_df.to_sql(TABLE_NAME_PREPROCESSED, engine, if_exists="replace", index=False)
        logging.info("Transformed dataset was successfully saved to db")
        
        # Predict pomocou uz nafitovaneho modelu a nasledne ulozenie zase do db
        logging.info('Preparing the data...')
        X_train, X_test, y_train, y_test = prepare_data(DATASET_PREPROCESSED)
        
        logging.info('Loading the model...')
        model = load_model(MODEL_NAME)

        logging.info('Calculating train dataset predictions...')
        y_pred_train = predict(X_train, model)
        logging.info('Calculating test dataset predictions...')
        y_pred_test = predict(X_test, model)

        # evaluate model
        logging.info('Evaluating the model...')
        train_error = mean_absolute_error(y_train, y_pred_train)
        test_error = mean_absolute_error(y_test, y_pred_test)

        logging.info('First 5 predictions:')
        logging.info(f'\n{X_test.head()}')
        logging.info(y_pred_test[:5])
        logging.info(f'Train error: {train_error}')
        logging.info(f'Test error: {test_error}')
        
        
        # Ulozenie predikcie do db (aj do csv)
        logging.info("Saving the predictions to db")
        prediction = pd.DataFrame(y_pred_test)
        prediction.to_sql(TABLE_NAME_PREDICTION, engine, if_exists="replace", index=False)
        prediction.to_csv(PREDICTIONS, index=False)
        logging.info("Predictions were successfully saved to db...")
                
    except Exception as e:
        logging.error(e)
        logging.error(traceback.format_exc())