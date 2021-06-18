

import matplotlib
import numpy as np
import pandas as pd
import torch
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
from flask import Flask, request, render_template, jsonify, url_for
from dataclasses import dataclass

from app.src.utils.utils import random_seed

matplotlib.use('Agg')
import warnings

# Quitar warnings innecesarios de la salida


warnings.filterwarnings('ignore')


MODEL = None
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# inicializar la app bajo el framework Flask
app = Flask(__name__)

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


def register_hook():
    save_output = SaveOutput()
    hook_handles = []

    for layer in MODEL.modules():
        if isinstance(layer, torch.nn.modules.conv.Conv2d):
            handle = layer.register_forward_hook(save_output)
            hook_handles.append(handle)
    return save_output

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
        elif request.form['tran_dash'] == 'dashboard':
            return render_template('dashboard.html')
        else:
            pass  # unknown
    elif request.method == 'GET':
        return render_template('contact.html')


    print('Admin das')


@app.route('/handle_data', methods=["GET", "POST"])
def handle_data():

    print('Recogemos los valores')

    ##Creamos un dataset para regoger los valores del formulario.
    df = pd.DataFrame()

    df['ciudad'] = request.form['ciudad']
    df['sexo'] = request.form['sexo']
    df['experiencia'] = request.form['experiencia']
    df['matricula'] = request.form['matricula']
    df['NivelEdu'] = request.form['NivelEdu']
    df['Educativo'] = request.form['Educativo']
    df['añosexperiencia'] = request.form['añosexperiencia']
    df['tamaño'] = request.form['tamaño']
    df['Sector'] = request.form['Sector']
    df['lastWork'] = request.form['lastWork']
    df['horas'] = request.form['horas']


    print(df.head)
    # your code
    return {'Recogiendo valores': 'Proyecto HIPF'}


# main
if __name__ == '__main__':
    # ejecución de la app

    ROOT_DIR1 = os.path.dirname(os.path.abspath(__file__))


    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG_MODE)

