import app as app
import flask
import matplotlib
import numpy as np
import pandas as pd
# import torch
import os
import random

from werkzeug.utils import redirect

import config
import base64
from utils import *
#from PIL import Image
from app import ROOT_DIR
from io import BytesIO
from app.src.models import train_model
#from app.src.models import  train
#from os import makedirs
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, jsonify, url_for, send_file, session
from dataclasses import dataclass

from app.src.models.predict import predict_pipeline, extrae, nlu, iniciar_nlu
from app.src.utils.utils import random_seed, plot_roc, plot_nlu
from app.src.data.conectBBDD import sql_table_train, sql_table_Predict, sql_connection, sql_table_nlu, \
    sql_Insert_predict, sql_update_predict, select_id, sql_insert_nlu, select_table, select_table_pred

matplotlib.use('Agg')
import warnings
import dash
import dash_table
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import matplotlib.pyplot as plt
import seaborn as sns
from app.src.utils.utils import plot_feature_vs_target, read_input, plot_predictions
from app.src.utils.utils_co import idh, latitude, longitude, city
from app.dashboard.layout import dashboard_layout, report_layout, historic_data_layout, model_layout, layout_map, \
    layout_general
from dash.dependencies import Input, Output, State
sns.set_theme()
import plotly.express as px
import plotly.graph_objects as go

from app.src.features.feature_engineering import DataFrameSelector, CreateFeatures, DropFeatures, Stringer, Imputer, Encoder
from app.src.features.pipeline import combine_features, buckets_experiencia
from sklearn.pipeline import Pipeline, FeatureUnion

# Quitar warnings innecesarios de la salida


warnings.filterwarnings('ignore')


MODEL = None

#DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# inicializar la app bajo el framework Flask
app = Flask(__name__)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
dash_app = dash.Dash(__name__, server=app, url_base_pathname='/admin_train/',
                     external_stylesheets=external_stylesheets)
dash_app.layout = html.Div(id='dash-container')
# On IBM Cloud Cloud Foundry, get the port number from the environment variable PORT
# When running this app on the local machine, default the port to 8000
port = int(os.getenv('PORT', 8001))


class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []


# def register_hook():
#     save_output = SaveOutput()
#     hook_handles = []
#
#     for layer in MODEL.modules():
#         if isinstance(layer, torch.nn.modules.conv.Conv2d):
#             handle = layer.register_forward_hook(save_output)
#             hook_handles.append(handle)
#     return save_output

#https://www.jetbrains.com/help/pycharm/creating-web-application-with-flask.html#login



@app.route('/')
def index():

    print('Conexion BBDD y creamos las tablas')
    con = sql_connection()
    sql_table_train(con)
    sql_table_Predict(con)
    sql_table_nlu(con)
    print('Tablas creadas')

###  ejemplo para MARCO para consultar tabla para graficar
    # query='SELECT * from nlu_hifp'
    # dftemp = pd.DataFrame(select_table(query))
    # print("Tabla pandas seleccionada nlu_hipf")
    # print(dftemp.iloc[-1])
    return render_template('Inicio.html')

input_values = read_input()
cities = input_values['ciudades']
idh = [idh(city) for city in cities]

@app.route('/login', methods=["GET", "POST"])
def login():


    print('Login')


    usuario = request.form['username']
    password = request.form['password']
    #request.session['usuario'] = usuario

    print(usuario)
    if usuario != 'admin':  # if a user is found, we want to redirect back to signup page so user can try again
       ##Accedemos al formulario
        return render_template('formu.html', cities=cities, idh=idh)
    else:
        ##Accedemos a la página de Administrador
        return render_template('signup_form.html')

@app.route('/logintrain', methods=["GET", "POST"])
def logintrain():

    usuario = 'admin'
       ##Accedemos al formulario
    return render_template('signup_form.html', cities=cities, idh=idh)


df = px.data.tips()
days = df.day.unique()
from app.src.utils.utils import df_for_plotting1, plot_roc, df_for_plotting
data_plot1 = df_for_plotting1()
data_plot = df_for_plotting()

@app.route('/admin_train', methods=["GET", "POST"])
def admin_train():

    print('admin_train')

    if request.method == 'POST':
        if request.form['tran_dash'] == 'train':
            print('---->Entrenamos el modelo')
            df_path = os.path.join(ROOT_DIR, 'data/ds_job.csv')
            print ('---->Ruta del fichero', df_path)
            train_model.training_pipeline(df_path)
            print('---->Sale del trainmodel')
            mensaje = 'El Modelo ha sido entrenado con Exito'

            return render_template('train.html', name=mensaje)

        elif request.form['tran_dash'] == 'dashboard':
            dashboard_layout(dash_app)
            #funcion3()
            #funcion1()
            #funcion2()
            return dash_app.index()

            #return render_template('dashboard.html', figure=fig)
        else:
            pass  # unknown
    elif request.method == 'GET':

        return render_template('contact.html')


@dash_app.callback(
              [Output('page-container', 'children'),
               Output('page-title', 'children')],
              [Input('btn-nclicks-1', 'n_clicks'),
              Input('btn-nclicks-2', 'n_clicks'),
              Input('btn-nclicks-3', 'n_clicks')]
              )
def displayClick(btn1, btn2, btn3):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'btn-nclicks-1' in changed_id:
        msg = 'Report de la encuesta realizada'
        return report_layout(), msg
    elif 'btn-nclicks-2' in changed_id:
        msg = 'Estudio de datos históricos de construcción del modelo'
        return historic_data_layout(), msg
    elif 'btn-nclicks-3' in changed_id:
        msg = 'Métricas del modelo en producción'
        return model_layout(), msg
    else:
        msg = 'Report de la encuesta realizada'
        return report_layout(), msg


@dash_app.callback(
              Output('container-1', 'children'),
              [Input('data_view', 'value')],
              )
def displayClick(value):
    if value == 'general':
        return layout_general()
    elif value == 'map':
        return layout_map()



@dash_app.callback(
    Output("bar-chart", "figure"),
    [Input("dropdown", "value")])
def update_bar_chart(col):
    #mask = df["day"] == day
    fig = plot_feature_vs_target(data_plot1[col], col)
    # fig = go.bar(data_plot[col], x="sex", y="total_bill",
    #              color="smoker", barmode="group")
    return fig

@dash_app.callback(
    Output("bar-chart-3", "figure"),
    [Input("dropdown-3", "value")])
def update_bar_chart(col):
    data_pred = select_table_pred()
    #mask = df["day"] == day
    #fig = plot_feature_vs_target(data_plot1[col], col)
    # fig = go.bar(data_plot[col], x="sex", y="total_bill",
    #              color="smoker", barmode="group")
    fig = plot_predictions(data_pred, col)
    return fig

@dash_app.callback(
    Output("bar-chart-circle", "figure"),
    [Input("dropdown-circle", "value")])
def update_bar_chart_circle(col):
    #mask = df["day"] == day
    X = data_plot[col].value_counts()
    #fig = go.Figure(data=[go.Pie(labels=X.index, values=X.values, hole=.3)])
    #X1 = pd.DataFrame([X.index, X.values], columns=['id', 'value'])
    #print(X1)
    X = pd.DataFrame(X).reset_index()
    X.columns = ['id', 'N']
    fig = px.pie(X, values='N', names='id', color_discrete_sequence=px.colors.sequential.Blues[::-1])
    fig.update_traces(textposition='inside', textinfo='percent+label')
    #fig = plot_feature_vs_target(data_plot[col], col)
    # fig = go.bar(data_plot[col], x="sex", y="total_bill",
    #              color="smoker", barmode="group")
    return fig

@dash_app.callback(
    Output("bar-chart-1", "figure"),
    [Input("dropdown-1", "value")])
def update_bar_chart_1(val):

    fig = plot_roc(val)
    # fig = go.bar(data_plot[col], x="sex", y="total_bill",
    #              color="smoker", barmode="group")
    return fig


@dash_app.callback(
    Output('bar-graph-nlu', 'figure'),
    [Input('buscar-id', 'n_clicks')],
    [State('input-on-submit', 'value')]
)
def update_nlu(n_clicks, value):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'n_clicks' in changed_id:
        print(value)
        value = int(value)
        print(value)
        data = select_table_pred()
        data.index = data.empleado_id.values
        ciudad = data.loc[value]['ciudad']
        print('ciudad: ', ciudad)

        new_data = data[data['ciudad'] == ciudad]
        print(new_data)
        emp = new_data['empleado_id'].values.tolist()
        print('emp: ', emp)
        new_data.index = emp
        print(new_data)
        emp.remove(value)
        print('new emp: ', emp)
        empleados = [value]
        if len(emp) == 0:
            pass
        elif len(emp) > 3:
            empleados += random.sample(emp, 3)
        else:
            empleados = empleados + emp
            print(empleados)

        fig = plot_nlu(empleados[::-1], ciudad)

        return fig
    else:
        fig = plot_nlu()
        return fig




@app.route('/handle_data', methods=["GET", "POST"])
def handle_data():

    print('Recogemos los valores...')


    #comentario = request.form['comen_sensa']

    features = pd.Series()


    id =select_id()

    print('valor ID...', id)

    features['empleado_id'] = id

    # if request.form['name_ciudad'] == "":
    #     features['ciudad'] = np.nan
    # else:
    #     features['ciudad'] = request.form['name_ciudad']
    features['ciudad'] = city(request.form['ciudad'])

    if request.form['ciudad'] == "":
        features['indice_desarrollo_ciudad'] = np.nan
    else:
        features['indice_desarrollo_ciudad'] = request.form['ciudad']

    if request.form['sexo'] == "":
        features['genero'] = np.nan
    else:
        features['genero'] = request.form['sexo']

    if request.form['experiencia'] == "":
        features['experiencia_relevante'] = np.nan
    else:
        features['experiencia_relevante'] = request.form['experiencia']

    if request.form['matricula']=="":
        features['universidad_matriculado'] =np.nan
    else:
        features['universidad_matriculado'] = request.form['matricula']

    if request.form['NivelEdu']=="":
        features['nivel_educacion'] =np.nan
    else:
        features['nivel_educacion'] = request.form['NivelEdu']

    if request.form['Educativo']=="":
        features['educacion'] =np.nan
    else:
        features['educacion'] = request.form['Educativo']


    if request.form['añosexperiencia']=="":
        features['experiencia'] =np.nan
    else:
        features['experiencia'] = request.form['añosexperiencia']


    features['tamano_compania'] = input_values['tamano_compania']
    features['tipo_compania'] = input_values['tipo_compania']
    # if request.form['tamaño']=="":
    #     features['tamano_compania'] =np.nan
    # else:
    #     features['tamano_compania'] = request.form['tamaño']
    #
    # if request.form['Sector']=="":
    #     features['tipo_compania'] =np.nan
    # else:
    #     features['tipo_compania'] = request.form['Sector']

    if request.form['lastWork']=="":
        features['ultimo_nuevo_trabajo'] =np.nan
    else:
        features['ultimo_nuevo_trabajo'] = request.form['lastWork']

    if request.form['horas'] == "":
        features['horas_formacion'] = np.nan
    else:
        features['horas_formacion'] = request.form['horas']

    print(features)


    #Ejecuta proceso de pipeline en prediccion
    y_pred = predict_pipeline(features)

    pred=y_pred[0]

    sql_Insert_predict(features)

    predict = pd.Series()
    predict['target'] = pred
    predict['empleado_id'] = id
    print('guardamos el resultado de la prediccion')

    print('modificamos la variable target ', predict)

    sql_update_predict(predict)

    predict = pd.Series()
    predict['target'] = id
    predict['empleado_id'] = id
    print('guardamos el resultado de la prediccion')

    #Cargamos las cajas string en un panda series con el id_ empleado
    pdnlu = pd.Series()

    pdnlu['empleado_id'] = id

    if request.form['comen_pago'] == "":
        pdnlu['pago'] = 'Neutro'
    else:
        pdnlu['pago'] = request.form['comen_pago']

    if request.form['comen_habilidad'] == "":
        pdnlu['habilidad'] = 'Neutro'
    else:
        pdnlu['habilidad'] = request.form['comen_habilidad']

    if request.form['comen_ambiente'] == "":
        pdnlu['ambiente'] = 'Neutro'
    else:
        pdnlu['ambiente'] = request.form['comen_ambiente']

    if request.form['comen_avance'] == "":
        pdnlu['avance'] = 'Neutro'
    else:
        pdnlu['avance'] = request.form['comen_avance']

    print("<< Mostrando los datos para analisis de sentimiento >>")
    print("   Empleado :",pdnlu[0])
    print("   Comentario por Pago :",pdnlu[1])
    print("   Comentario por habilidad :",pdnlu[2])
    print("   Comentario por ambiente :",pdnlu[3])
    print("   Comentario por avance :",pdnlu[4])

    # con el string de comentario incluimos la llamada a la funcion de analisis de sentimiento
    natural_language_understanding = iniciar_nlu()
    #score_nlu = pd.Series([0.19,-0.3,-0.5,0.21])
    score_nlu = [extrae(x, 0, natural_language_understanding) for x in pdnlu[1:5]]

    print(score_nlu)
    for i,scores in enumerate(score_nlu):
        print("\n### COMENTARIO ", i+1)
        print(pdnlu[i+1])
        if scores > 0.2:
            print("\nEste comentario tiene un sentimiento positivo con NLU {0}%".format(scores*100))
        elif (scores>0 and scores<0.2):
            print("\nEste comentario tiene un sentimiento neutro/no positivo con NLU {0}%".format(scores*100))
        else:
            print("\nEste comentario tiene un sentimiento negativo con NLU {0}%".format(scores*100))


    ####Aquí llamamos a una funcion para insertar los datos en la tabla nlu_hifp
    #print(pdnlu[0], pdnlu[1], pdnlu[2], pdnlu[3], pdnlu[4], score_nlu[0], score_nlu[1], score_nlu[2], score_nlu[3])
    sql_insert_nlu(pdnlu, score_nlu)
    print("\n### Los datos fueron insertados en tabla Nlu_hipf ### ")

    mensaje='Muchas gracias por enviar su encuesta.'

    return render_template('encuesta_fin.html', name=mensaje)


# main
if __name__ == '__main__':
    # ejecución de la app

    ROOT_DIR1 = os.path.dirname(os.path.abspath(__file__))

    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG_MODE)

