import flask
import matplotlib
import numpy as np
import pandas as pd
# import torch
import os

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
from flask import Flask, request, render_template, jsonify, url_for, send_file
from dataclasses import dataclass

from app.src.models.predict import predict_pipeline
from app.src.utils.utils import random_seed

matplotlib.use('Agg')
import warnings


import dash
import dash_table
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import matplotlib.pyplot as plt
import seaborn as sns
from app.src.utils.utils import plot_feature_vs_target
from app.dashboard.layout import dashboard_layout
from dash.dependencies import Input, Output
sns.set_theme()
import plotly.express as px


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
port = int(os.getenv('PORT', 8000))


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
    return render_template('Inicio.html')



@app.route('/login', methods=["GET", "POST"])
def login():
    usuario = request.form['username']
    password = request.form['password']

    if usuario != 'admin':  # if a user is found, we want to redirect back to signup page so user can try again
       ##Accedemos al formulario
        return render_template('formu.html')
    else:
        ##Accedemos a la página de Administrador
        return render_template('signup_form.html')


df = px.data.tips()
days = df.day.unique()
from app.src.utils.utils import df_for_plotting1
data_plot = df_for_plotting1()

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
            return render_template('train.html')
            #return render_template('signup_form.html')
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

def funcion2():
    colors = {
        'background': '#111111',
        'text': '#7FDBFF'
    }
    df = pd.DataFrame({
        "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
        "Amount": [4, 1, 2, 2, 4, 5],
        "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
    })

    fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")
    dash_app.update_layout = html.Div(style={'backgroundColor': colors['background']}, children=[
        html.H1(
            children='Hello Dash',
            style={
                'textAlign': 'center',
                'color': colors['text']
            }
        ),

        html.Div(children='Dash: A web application framework for Python.', style={
            'textAlign': 'center',
            'color': colors['text']
        }),

        dcc.Graph(
            id='example-graph-2',
            figure=fig
        )
    ])

def funcion1():
    dash_app.layout = html.Div([
                    dcc.Dropdown(
                        id="dropdown",
                        options=[{"label": x, "value": x} for x in data_plot.keys()],
                        value=list(data_plot.keys())[0],
                        clearable=False,
                    ),
                    dcc.Graph(id="bar-chart"),
                ])

def funcion3():
    df = pd.DataFrame({
        "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
        "Amount": [4, 1, 2, 2, 4, 5],
        "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
    })

    fig1 = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")
    dash_app.layout = html.Div(children=[
        html.Div(children=[
        html.H1(children='HIPF DASHBOARD', className='header-title'),

        html.P(
            children="Analiza la fuga de empleados de forma simple y efectiva",
            className="header-description",
        ),
            ], className='header'),

        html.Div([
            html.Div([
            dash_table.DataTable(
                id='table',
                columns=[{"name": i, "id": i} for i in df.columns],
                data=df.to_dict('records'),
                style_table={'width': '50%'}

            )], className='quarter'),
            html.Div([
                dcc.Graph(
                    id='example-graph-2',
                    figure=fig1
                ),
            ], className='quarter')
        ], className='row'),


        html.Div([
            html.Div([
                dcc.Dropdown(
                    id="dropdown",
                    options=[{"label": x, "value": x} for x in data_plot.keys()],
                    value=list(data_plot.keys())[0],
                    clearable=False,
                ),
                dcc.Graph(id="bar-chart"),
            ], className='quarter'),
            html.Div([
                dcc.Graph(
                    id='example-graph-2',
                    figure=fig1
                ),
            ], className='quarter')
        ], className='row')
    ])



@dash_app.callback(
    Output("bar-chart", "figure"),
    [Input("dropdown", "value")])
def update_bar_chart(col):
    #mask = df["day"] == day
    fig = plot_feature_vs_target(data_plot[col], col)
    # fig = go.bar(data_plot[col], x="sex", y="total_bill",
    #              color="smoker", barmode="group")
    return fig

data = pd.read_csv('app/data/ds_job.csv')[:50]
#print(data)
@dash_app.callback(
    [Output("table", "columns"), Output("table", "data")],
    [Input("drop_table", "value")])
def update_table(col):
    # mask = df["day"] == day
    print('hola', col)
    if col != 'nada':
        new_data = data[[col, 'target','empleado_id']]
        new_columns = [col, 'target', 'N']
        cols_to_group = [col, 'target']
        new_data = new_data.groupby(cols_to_group).count()
        new_data.columns = ['N']
        return new_data.to_dict('records'), ['N']
    else:
        return data.to_dict('records'), data.columns

df = pd.DataFrame({
        "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
        "Amount": [4, 1, 2, 2, 4, 5],
        "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
    })


print('Admin das')

@app.route('/fig')
def fig(text='hola'):
    from io import BytesIO
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(range(10), 'b-')
    plt.title=text

    img=BytesIO()
    fig.savefig(img)
    img.seek(0)
    return send_file(img, mimetype='image/png')


@app.route('/handle_data', methods=["GET", "POST"])
def handle_data():

    print('Recogemos los valores para un registro de empleado')

    ##Creamos un dataset para recoger los valores del formulario.

    print('Recogemos los valores...', )

    features = [[0,request.form['name_ciudad'],
                 request.form['ciudad'],
                 request.form['sexo'],
                 request.form['experiencia'],
                 request.form['matricula'],
                 request.form['NivelEdu'],
                 request.form['Educativo'],
                 request.form['añosexperiencia'],
                 request.form['tamaño'],
                 request.form['Sector'],
                 request.form['lastWork'],
                 request.form['horas'],
                 request.form['comen_sensa']]]

    y_pred = predict_pipeline(features)

    # con el string de comentario incluimos la llamada a la funcion de analisis de sentimiento
    score_nlu = extrae(comen_sensa, 0)
    if score_nlu > 0:
        print("Score del comentario positivo con NLU {0}%".format(score_nlu*100))
    else:
        print("Score del comentario negativo con NLU {0}%".format(score_nlu*100))

    return {'Predicted value': y_pred}


# main
if __name__ == '__main__':
    # ejecución de la app

    ROOT_DIR1 = os.path.dirname(os.path.abspath(__file__))


    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG_MODE)

