from cloudant.client import Cloudant
import ibm_boto3
from ibm_botocore.client import Config
from ibm_botocore.client import ClientError
import pickle
from io import BytesIO

import time
#import torch
import random
import functools
import numpy as np
from typing import Any, Callable, TypeVar, cast

from.utils_co import latitude, longitude
from app.src.data.conectBBDD import select_table_pred, select_table

from.utils_co import latitude, longitude


def random_seed(seed_value: int) -> None:
    """
    Random Seeds Numpy, Random and Torch libraries
    Args:
        seed_value (int): Number for seeding
    """
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu vars
    random.seed(seed_value)  # Python
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False


F = TypeVar('F', bound=Callable[..., Any])


def timer(func: F) -> F:
    """ Print the runtime of the decorated function """
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        _ = time.perf_counter() - start_time
        hours, _ = divmod(_, 3600)
        minutes, seconds = divmod(_, 60)

        print(f'Execution time of function {func.__name__!r}: {hours:.0f} hrs {minutes:.0f} mins {seconds:.3f} secs')
        return value
    return cast(F, wrapper_timer)





class DocumentDB:
    """
        Clase para gestionar la base de datos documental IBM Cloudant
    """

    def __init__(self, username, api_key):
        """
            Constructor de la conexión a IBM cloudant
            Args:
               username (str): usuario.
               apikey (str): API key.
        """
        self.connection = Cloudant.iam(username, api_key, connect=True)
        self.connection.connect()

    def get_database(self, db_name):
        """
            Función para obtener la base de datos elegida.
            Args:
               db_name (str):  Nombre de la base de datos.
            Returns:
               Database. Conexión a la base de datos elegida.
        """
        return self.connection[db_name]

    def database_exists(self, db_name):
        """
            Función para comprobar si existe la base de datos.
            Args:
               db_name (str):  Nombre de la base de datos.
            Returns:
               boolean. Existencia o no de la base de datos.
        """
        return self.get_database(db_name).exists()

    def create_document(self, db, document_dict):
        """
            Función para crear un documento en la base de datos
            Args:
               db (str):  Conexión a una base de datos.
               document_dict (dict):  Documento a insertar.
        """
        db.create_document(document_dict)


class IBMCOS:
    """
        Clase para gestionar el repositorio de objetos IBM COS
    """

    def __init__(self, ibm_api_key_id, ibm_service_instance_id, ibm_auth_endpoint, endpoint_url):
        """
            Constructor de la conexión a IBM COS
            Args:
               ibm_api_key_id (str): API key.
               ibm_service_instance_id (str): Service Instance ID.
               ibm_auth_endpoint (str): Auth Endpoint.
               endpoint_url (str): Endpoint URL.
        """


        self.connection = ibm_boto3.resource("s3",
                                             ibm_api_key_id=ibm_api_key_id,
                                             ibm_service_instance_id=ibm_service_instance_id,
                                             ibm_auth_endpoint=ibm_auth_endpoint,
                                             config=Config(signature_version="oauth"),
                                             endpoint_url=endpoint_url)

    def save_object_in_cos(self, obj, name, timestamp, bucket_name='hipf-models-2'):
        """
            Función para guardar objeto en IBM COS.
            Args:
               obj:  Objeto a guardar.
               name (str):  Nombre del objeto a guardar.
               timestamp (float): Segundos transcurridos.
            Kwargs:
                bucket_name (str): depósito de COS elegido.
        """

        # objeto serializado
        pickle_byte_obj = pickle.dumps(obj)
        # nombre del objeto en COS
        pkl_key = name + "_" + str(int(timestamp)) + ".pkl"

        try:
            # guardado del objeto en COS
            self.connection.Object(bucket_name, pkl_key).put(
                Body=pickle_byte_obj
            )
        except ClientError as be:
            print("CLIENT ERROR: {0}\n".format(be))
        except Exception as e:
            print("Unable to create object: {0}".format(e))

    def get_object_in_cos(self, key, bucket_name='hipf-models-2'):
        """
            Función para obtener un objeto de IBM COS.
            Args:
               key (str):  Nombre del objeto a obtener de COS.
            Kwargs:
                bucket_name (str): depósito de COS elegido.
            Returns:
               obj. Objeto descargado.
        """

        # conexión de E/S de bytes
        with BytesIO() as data:
            # descarga del objeto desde COS
            self.connection.Bucket(bucket_name).download_fileobj(key, data)
            data.seek(0)
            # des-serialización del objeto descargado
            obj = pickle.load(data)
        return obj

import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
def plot_feature_vs_target(df, column, vals=None):

    if column == 'experiencia':
        vals = ['<1'] + list(range(1, 21)) + ['>20']
        vals = [str(i) for i in vals]
    elif column  == 'nivel_educacion':
        vals = ['Primary School', 'High School', 'Graduate', 'Masters', 'Phd']
    elif column == 'ultimo_nuevo_trabajo':
        vals = ['never', '1', '2', '3', '4', '>4']
    else:
        vals = np.unique(df[column])
    # if vals is None:
    #     vals = np.unique(df[column])
    t = np.zeros((2, len(vals)))
    for i, value in enumerate(vals):

        t[0, i] = df[(df[column] == value) & (df['target'] == 0.0)].value_counts().values[0]
        t[1, i] = df[(df[column] == value) & (df['target'] == 1.0)].value_counts().values[0]
        t[0,i] /= df[df[column] == value].value_counts().sum()
        t[1,i] /= df[df[column] == value].value_counts().sum()

    df1 = pd.DataFrame(t.T, index=vals).reset_index()
    df1.columns = ['vals', 0, 1]
    print(df1)
    fig = go.Figure(data=[
        go.Bar(name='0', x=df1.index, y=df1[0], marker_color='rgb(55, 83, 109)'),
        go.Bar(name='1', x=df1.index, y=df1[1], marker_color='rgb(26, 118, 255)')
    ])
    # plt.figure()
    # df1.plot.bar(stacked=True)
    # plt.title('Target por ' + column)
    # plt.show()
    fig.update_layout(barmode='stack',
                      xaxis_title=column,
                      yaxis_title='Porcentaje'
                      )
    fig.update_xaxes(
        tickvals=df1.index,
        ticktext=df1['vals'].values)
    #fig.show()
    return fig

data = pd.read_csv('app/data/ds_job.csv')
from ..features.feature_engineering import buckets_numericos, buckets_experiencia, buckets_tam_compania
def df_for_plotting1():
    # Preparamos un nuevo diccionario con los DF para plottear, creando cada
    # columna individualmente
    new_data = {}
    test = data[['indice_desarrollo_ciudad', 'target']]
    test['id_desarrollo_ciudad'] = test.indice_desarrollo_ciudad.apply(buckets_numericos)
    test['id_desarrollo_ciudad'] = test['id_desarrollo_ciudad'].apply(lambda x: str(x))
    test = test.drop('indice_desarrollo_ciudad', axis=1)
    new_data['id_desarrollo_ciudad'] = test

    test = data[['target', 'experiencia']]
    test['experiencia'] = test['experiencia'].apply(lambda x: str(x))
    new_data['experiencia'] = test

    test = data[['target', 'experiencia_relevante']]
    test['experiencia_relevante'] = test['experiencia_relevante'].apply(lambda x: str(x))
    new_data['experiencia_relevante'] = test[test['experiencia_relevante'] != 'nan']

    test = data[['target', 'universidad_matriculado']]
    test['universidad_matriculado'] = test['universidad_matriculado'].apply(lambda x: str(x))
    new_data['universidad_matriculado'] = test[test['universidad_matriculado'] != 'nan']


    test = data[['target', 'ultimo_nuevo_trabajo']]
    test['ultimo_nuevo_trabajo'] = test['ultimo_nuevo_trabajo'].apply(lambda x: str(x))
    new_data['ultimo_nuevo_trabajo'] = test[test['ultimo_nuevo_trabajo'] != 'nan']

    test = data[['target', 'genero']]
    test['genero'] = test['genero'].apply(lambda x: str(x))
    new_data['genero'] = test[test['genero'] != 'nan']

    test = data[['target', 'nivel_educacion']]
    test['nivel_educacion'] = test['nivel_educacion'].apply(lambda x: str(x))
    new_data['nivel_educacion'] = test[test['nivel_educacion'] != 'nan']

    test = data[['target', 'educacion']]
    test['educacion'] = test['educacion'].apply(lambda x: str(x))
    new_data['educacion'] = test[test['educacion'] != 'nan']

    test = data[['target', 'tamano_compania']]
    test['tamano_compania'] = test['tamano_compania'].apply(lambda x: str(x))
    new_data['tamano_compania'] = test[test['tamano_compania'] != 'nan']

    test = data[['target', 'tipo_compania']]
    test['tipo_compania'] = test['tipo_compania'].apply(lambda x: str(x))
    new_data['tipo_compania'] = test[test['tipo_compania'] != 'nan']

    return new_data


def df_for_plotting():
    df = data.drop(columns=['ciudad', 'empleado_id'])
    df['indice_desarrollo_ciudad'] = df['indice_desarrollo_ciudad'].apply(buckets_numericos)
    df['horas_formacion'] = df['horas_formacion'].apply(lambda x: buckets_numericos(x, limits=[10, 20, 50, 100]))
    df['experiencia'] = df['experiencia'].apply(buckets_experiencia)
    #df['tamano_compania'] = df['tamano_compania'].apply(buckets_tam_compania)
    return df


def plot_roc(val):
    name = 'model_roc_' + val + '.pkl'
    path = 'app/dashboard/objects/' + name
    with open(path, 'rb') as f:
        df = pickle.load(f)

    fig = px.line(df, x='fpr', y='tpr', color='model', labels={
                     "fpr": "False Positive Rate",
                     "tpr": "True Positive Rate"
                 }, title='ROC Curves of different models', hover_data=['AUC_Score'])
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))
    return fig

def plot_predictions(data, col):
    data = data[[col, 'target', 'empleado_id']]
    data1 = data[data['target'] == 1.0]
    data2 = data[data['target'] == 0.0]
    data1 = data1.groupby([col, 'target']).count().reset_index()
    data2 = data2.groupby([col, 'target']).count().reset_index()
    data3 = pd.concat([data1, data2])
    data3.columns = [col, 'prediction', 'N']
    data3.prediction = data3.prediction.map({0.0: 'Negative', 1.0: 'Positive'})
    fig = px.bar(data3, x=col, y='N', color="prediction", barmode='group',
                 color_discrete_map={
                     'Negative': 'rgb(55, 83, 109)',
                     'Positive': 'rgb(26, 118, 255)'
                 }
                 )
    return fig

def df_for_map(data):
    data = data[['ciudad', 'target', 'empleado_id']]
    data = data[data['target'] == 1.0]
    data['lat'] = data['ciudad'].apply(latitude)
    data['lon'] = data['ciudad'].apply(longitude)
    data = data.drop(['target'], axis=1).groupby(['ciudad', 'lat', 'lon']).count().reset_index()
    data.columns = ['ciudad', 'latitude', 'longitude', 'N']
    return data

def read_input():
    vars = {}
    with open('input.txt', 'r') as f:
        #input_data = f.read()
        for line in f:
            # Si la línea empieza por almohadilla lo consideramos comentario
            # Si no, separamos por el signo = y guardamos el texto a la izquierda del =
            #   como key del diccionario y el texto de la derecha como value
            if line.strip()[0] != '#':
                vals = line.strip().split('=')
                # En caso de que haya comas separando en el value, lo cargamos como lista
                if ',' not in vals[1]:
                    vars[vals[0].strip()] = vals[1].strip()
                else:
                    lista = vals[1].split(',')
                    vars[vals[0].strip()] = [x.strip() for x in lista]
        return vars

def plot_predictions(data, col):
    data = data[[col, 'target', 'empleado_id']]
    data1 = data[data['target'] == 1.0]
    data2 = data[data['target'] == 0.0]
    data1 = data1.groupby([col, 'target']).count().reset_index()
    data2 = data2.groupby([col, 'target']).count().reset_index()
    data3 = pd.concat([data1, data2])
    data3.columns = [col, 'prediction', 'N']
    data3.prediction = data3.prediction.map({0.0: 'Negative', 1.0: 'Positive'})
    fig = px.bar(data3, x=col, y='N', color="prediction", barmode='group',
                 color_discrete_map={
                     'Negative': 'rgb(55, 83, 109)',
                     'Positive': 'rgb(26, 118, 255)'
                 }
                 )
    return fig

def df_for_map(data):
    data = data[['ciudad', 'target', 'empleado_id']]
    data = data[data['target'] == 1.0]
    data['lat'] = data['ciudad'].apply(latitude)
    data['lon'] = data['ciudad'].apply(longitude)
    data = data.drop(['target'], axis=1).groupby(['ciudad', 'lat', 'lon']).count().reset_index()
    data.columns = ['ciudad', 'latitude', 'longitude', 'N']
    return data

def plot_nlu(empleados=None, ciudad=None):
    query = 'SELECT * from nlu_hifp'
    dftemp = pd.DataFrame(select_table(query))[[0, 5, 6, 7, 8]]
    dftemp.columns = ['empleado_id', 'sc_pago', 'sc_habilidad', 'sc_ambiente', 'sc_avance']
    if empleados is not None:
        dftemp.index = dftemp.empleado_id.values
        dftemp = dftemp.loc[empleados]
    else:
        dftemp = dftemp.iloc[:4]

    fig2 = go.Figure(data=[
        go.Bar(name='sc_pago', x=dftemp.sc_pago, marker_color='rgb(55, 83, 109)'),
        go.Bar(name='sc_habilidad', x=dftemp.sc_habilidad, marker_color='rgb(26, 118, 255)'),
        go.Bar(name='sc_ambiente', x=dftemp.sc_ambiente, marker_color='rgb(78, 123, 212)'),
        go.Bar(name='sc_avance', x=dftemp.sc_avance, marker_color='rgb(152, 225, 213)')
    ])
    if ciudad is not None:
        fig2.update_layout(
            title='Score de sentimiento en función de distintos aspectos de empleados de la sede de ' + ciudad,
            xaxis_title="Score",
            yaxis_title="ID empleado",
        )

    else:
        fig2.update_layout(title='Score de sentimiento de empleados en función de distintos aspectos',
                           xaxis_title="Score",
                           yaxis_title="ID empleado",
                           )
    fig2.update_xaxes(range=[-1, 1])
    fig2.update_yaxes(
        tickvals=list(range(len(dftemp))),
        ticktext=dftemp['empleado_id'].values
    )
    return fig2

def read_input():
    vars = {}
    with open('input.txt', 'r') as f:
        #input_data = f.read()
        for line in f:
            # Si la línea empieza por almohadilla lo consideramos comentario
            # Si no, separamos por el signo = y guardamos el texto a la izquierda del =
            #   como key del diccionario y el texto de la derecha como value
            if line.strip()[0] != '#':
                vals = line.strip().split('=')
                # En caso de que haya comas separando en el value, lo cargamos como lista
                if ',' not in vals[1]:
                    vars[vals[0].strip()] = vals[1].strip()
                else:
                    lista = vals[1].split(',')
                    vars[vals[0].strip()] = [x.strip() for x in lista]
        return vars
