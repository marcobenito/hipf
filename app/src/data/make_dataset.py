import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from app import cos

from sklearn.preprocessing import MinMaxScaler
from ..features.feature_engineering import DataFrameSelector, CreateFeatures, Stringer, DropFeatures, Imputer, Encoder
from ..features.pipeline import combine_features, buckets_experiencia
import pickle

def make_dataset(path, timestamp):

    """
        Función que permite crear el dataset usado para el entrenamiento
        del modelo.
        Args:
           path (str):  Ruta hacia los datos.
           timestamp (float):  Representación temporal en segundos.
           target (str):  Variable dependiente a usar.
        Kwargs:
           model_type (str): tipo de modelo usado.
        Returns:
           DataFrame, DataFrame. Datasets de train y test para el modelo.
    """

    print('---> Getting data')
    df = get_raw_data_from_local(path)
    return df.copy()


def split_dataset(df):
    # Separamos la variable objetivo
    y = df['target']
    X = df.drop('target', axis=1)

    print('---> Train / test split')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    return X_train.copy(), X_test.copy(),y_train.copy(), y_test.copy()

def make_RandomOverSampler(X_train, y_train):

    # Aplicamos el oversampling
    os = RandomOverSampler(random_state=1, sampling_strategy=0.5)
    X_train_res, y_train1 = os.fit_resample(X_train, y_train)

    return X_train_res.copy(), y_train1.copy()



def get_raw_data_from_local(path):

    """
        Función para obtener los datos originales desde local
        Args:
           path (str):  Ruta hacia los datos.
        Returns:
           DataFrame. Dataset con los datos de entrada.
    """

    df = pd.read_csv(path)
    return df.copy()


def get_raw_pikle_from_local(path):

    """
        Función para obtener el pickle con los mejores parametros
        Args:
           path (str):  Ruta hacia los datos.
        Returns:
           DataFrame. Dataset con los datos de entrada.
    """


    with open(path, 'rb') as f:
        params = pickle.load(f)


    return params
