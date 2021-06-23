import pandas as pd
#from ..features.feature_engineering import feature_engineering
from app import cos, init_cols


def make_dataset(data, model_info):

    """
        Función que permite crear el dataset usado para el entrenamiento
        del modelo.

        Args:
           data (List):  Lista con la observación llegada por request.
           model_info (dict):  Información del modelo en producción.

        Kwargs:
           model_type (str): tipo de modelo usado.

        Returns:
           DataFrame. Dataset a inferir.
    """

    print('---> Getting data')
    data_df = get_raw_data_from_request(data)

    print('data_df',data_df)


    ## print('---> Transforming data')
    ##data_df = transform_data(data_df, model_info)
    ##print('---> Feature engineering')
    ##data_df = feature_engineering(data_df)
    ##print('---> Preparing data for training')
    ## data_df = pre_train_data_prep(data_df, model_info)

    return data_df.copy()


def get_raw_data_from_request(data):

    """
        Función para obtener nuevas observaciones desde request

        Args:
           data (List):  Lista con la observación llegada por request.

        Returns:
           DataFrame. Dataset con los datos de entrada.
    """
    df = pd.DataFrame(data,columns=init_cols)
    df.reset_index(drop=True, inplace=True)
    return df.copy()


def transform_data(data_df, model_info):
    """
        Función que permite realizar las primeras tareas de transformación
        de los datos de entrada.

        Args:
            data_df (DataFrame):  Dataset de entrada.
            model_info (dict):  Información del modelo en producción.
            cols_to_remove (list): Columnas a retirar.

        Returns:
           DataFrame. Dataset transformado.
    """

    print('------> Removing unnecessary columns')
    #data_df = remove_unwanted_columns(data_df, cols_to_remove)



    # creando dummies originales
    print('------> Encoding data')
    print('---------> Getting encoded columns from cos')
    enc_key = model_info['objects']['encoders']+'.pkl'
    print(' enc_key-->', enc_key)
    # obteniendo las columnas presentes en el entrenamiento desde COS
    enc_cols = cos.get_object_in_cos(enc_key)
    print(' enc_cols-->', enc_cols)
    # columnas dummies generadas en los datos de entrada
    data_df = pd.get_dummies(data_df)

    # agregando las columnas dummies faltantes en los datos de entrada
    data_df = data_df.reindex(columns=enc_cols, fill_value=0)

    return data_df.copy()


def pre_train_data_prep(data_df, model_info):

    """
        Función que realiza las últimas transformaciones sobre los datos
        antes del entrenamiento (imputación de nulos)

        Args:
            data_df (DataFrame):  Dataset de entrada.
            model_info (dict):  Información del modelo en producción.

        Returns:
            DataFrame. Datasets de salida.
    """

    print('------> Getting imputer from cos')
    imputer_key = model_info['objects']['imputer']+'.pkl'
    data_df = input_missing_values(data_df, imputer_key)

    return data_df.copy()


def input_missing_values(data_df, key):

    """
        Función para la imputación de nulos

        Args:
            data_df (DataFrame):  Dataset de entrada.
            key (str):  Nombre del objeto imputador en COS.

        Returns:
            DataFrame. Datasets de salida.
    """

    print('------> Inputing missing values')
    # obtenemos el objeto SimpleImputer desde COS
    imputer = cos.get_object_in_cos(key)
    data_df = pd.DataFrame(imputer.transform(data_df), columns=data_df.columns)

    return data_df.copy()


def remove_unwanted_columns(df, cols_to_remove):
    """
        Función para quitar variables innecesarias

        Args:
           df (DataFrame):  Dataset.

        Returns:
           DataFrame. Dataset.
    """
    return df.drop(columns=cols_to_remove)





