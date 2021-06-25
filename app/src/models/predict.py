import pickle

from sklearn.preprocessing import MinMaxScaler

from ..data.make_datasetPredict import make_dataset
from app import cos, client
from cloudant.query import Query
# Librerias para incluir NLU
import json
import ibm_watson
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 import Features, CategoriesOptions, RelationsOptions, EmotionOptions
from ibm_watson.natural_language_understanding_v1 import SentimentOptions
import pandas as pd
import numpy as np
#Inicializar datos de acceso desde el script que evaluará los datos de comentarios
# authenticator = IAMAuthenticator(iam_authentic)
# natural_language_understanding = NaturalLanguageUnderstandingV1(
#     version = nlu_version,
#     authenticator = authenticator
# )
#
# natural_language_understanding.set_service_url(url_service)

def predict_pipeline(data, model_info_db_name='hipf_db'):

    """
        Función para gestionar el pipeline completo de inferencia
        del modelo.
        Args:
            path (str):  Ruta hacia los datos.
        Kwargs:
            model_info_db_name (str):  base de datos a usar para almacenar
            la info del modelo.
        Returns:
            list. Lista con las predicciones hechas.
    """

    # Carga de la configuración de entrenamiento
    model_config = load_model_config(model_info_db_name)['model_config']

    from ..features.feature_engineering import DataFrameSelector, CreateFeatures, DropFeatures, Stringer, Imputer, Encoder
    from ..features.pipeline import combine_features, buckets_experiencia
    from sklearn.pipeline import Pipeline, FeatureUnion


    # obteniendo la información del modelo en producción
    model_info = get_best_model_info(model_info_db_name)
    # cargando y transformando los datos de entrada
    #data_df = make_dataset(data, model_info)


    with open('checkpoint/pipeline.pkl', 'rb') as f:
        pipe = pickle.load(f)

    X_pred = pipe.transform(data)

    # Descargando el objeto del modelo
    model_name = model_info['name']+'.pkl'
    print('------> Loading the model {} object from the cloud'.format(model_name))
    model = load_model(model_name)
    #print('model-->',model)

    #print(X_pred)
    #print(X_pred.shape)
    print('-----------------------prediccion->',model.predict(X_pred)[0])


    # realizando la inferencia con los datos de entrada
    return model.predict(X_pred).tolist()

def load_model(name, bucket_name='models-hifp'):
    """
         Función para cargar el modelo en IBM COS

         Args:
             name (str):  Nombre de objeto en COS a cargar.

         Kwargs:
             bucket_name (str):  depósito de IBM COS a usar.

        Returns:
            obj. Objeto descargado.
     """
    return cos.get_object_in_cos(name, bucket_name)


def get_best_model_info(db_name):
    """
         Función para cargar la info del modelo de IBM Cloudant

         Args:
             db_name (str):  base de datos a usar.

         Kwargs:
             bucket_name (str):  depósito de IBM COS a usar.

        Returns:
            dict. Info del modelo.
     """
    db = client.get_database(db_name)
    query = Query(db, selector={'status': {'$eq': 'in_production'}})
    return query()['docs'][0]


def load_model_config(db_name):
    """
        Función para cargar la info del modelo desde IBM Cloudant.

        Args:
            db_name (str):  Nombre de la base de datos.

        Returns:
            dict. Documento con la configuración del modelo.
    """
    db = client.get_database(db_name)
    query = Query(db, selector={'_id': {'$eq': 'model_config'}})
    return query()['docs'][0]

def iniciar_nlu():
    # Inicializar datos de acceso desde el script que evaluara los datos de comentarios
    iam_authentic = 'Ruk59D8Cw_hwNNdl7KoyK5uEtZVNVt_C5Xx9W-uoG-BN'
    nlu_version = '2019-07-12'
    url_service = 'https://api.us-south.natural-language-understanding.watson.cloud.ibm.com/instances/70cf1ea3-87d2-43b8-9e1b-a02aab274d75'
    tipo_analisis = "sentiment"
    authenticator = IAMAuthenticator(iam_authentic)
    natural_language_understanding = NaturalLanguageUnderstandingV1(
        version=nlu_version,
        authenticator=authenticator
    )

    natural_language_understanding.set_service_url(url_service)
    return natural_language_understanding


def nlu(texto,     natural_language_understanding ):
    """
        Definimos una funcion para llamada al analisis de sentimiento enviandole una frase
        que va imprimiendo cada resultado por frase que se va enviando
        Args:
            texto (str) : variable que trae la frase a evaluar utilizando la funcion de NLU
                    de Watson con la funcion de libreria natural_language_undestanding
            natural_language_understanding : variable que trae el servicio de NLU cargado en la funcion iniciar_nlu()
        Returns :
            response es una lista json con el contenido devuelto por analisis de sentimiento
                    con score y con el label entregado por el analisis evaluado
    """

    response = natural_language_understanding.analyze(
        text=texto,
        language='es',
        features=Features(sentiment=SentimentOptions())
    ).get_result()

    return response


# Definimos funcion para ejecutar la carga de resultados en un dataframe jsonlist
def extrae(comentario, tipo_res,    natural_language_understanding ):
    """
        Definimos una funcion para llamada al analisis de sentimiento enviandole una frase
        que va analizando y guardando en una lista cada resultado por frase que se va enviando
        Args:
        comentario : variable que trae la columna de comentario sobre el string a evaluar utilizando la funcion de NLU
                    de Watson con la funcion construida nlu
        tipo_res : variable que trae dos valores
                0 : Para que extraiga solo el score
                1 : Para que extraiga label y score en formato json
        natural_language_understanding : variable que trae el servicio de NLU cargado en la funcion iniciar_nlu()
        returns :
            jsonlist es un dataframe convertido de lista json con el contenido devuelto por analisis de sentimiento
                    con score y con el label entregado por el analisis evaluado de cada frase
    """
    tipo_analisis = "sentiment"
    jsonlist = pd.DataFrame()
    responses = nlu(comentario,    natural_language_understanding )
    respuesta = responses[tipo_analisis]
    jsonlist = pd.DataFrame({'label': [respuesta["document"]["label"]], 'score': [respuesta["document"]["score"]]})
    print(jsonlist)
    if tipo_res == 0:
        return np.array(jsonlist.score)
    elif tipo_res == 1:
        return jsonlist

