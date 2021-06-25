import dash_html_components as html
import dash_core_components as dcc
#import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import dash
import dash_table
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from app.src.utils.utils import df_for_plotting1, df_for_plotting
from app.src.models.train_model import load_model_metrics



def dashboard_layout(dash_app):
    dash_app.layout = html.Div(children=[
        html.Div(children=[
            html.H1(children='HIPF DASHBOARD', className='header-title'),

            # html.P(
            #     children="Analiza la fuga de empleados de forma simple y efectiva",
            #     className="header-description",
            # ),


        ], className='header'),
        html.Div([
            html.Button('Resultado encuesta', id='btn-nclicks-1', n_clicks=0, className='button'),
            html.Button('Datos históricos', id='btn-nclicks-2', n_clicks=0, className='button'),
            html.Button('Modelo', id='btn-nclicks-3', n_clicks=0, className='button')
        ], className='button-row'),
        html.H1(id='page-title'),
        html.Div(id='page-container')
        ])

def report_layout():
    df = pd.DataFrame({
        "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
        "Amount": [4, 1, 2, 2, 4, 5],
        "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
    })
    data = pd.read_csv('app/data/ds_job.csv')[:50]
    data_1 = pd.read_csv('app/data/new_data.csv')
    data = data[['empleado_id', 'ciudad', 'genero', 'experiencia', 'universidad_matriculado', 'target']]
    data_plot = df_for_plotting1()
    fig1 = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")
    print(data_1)
    fig1 = px.scatter_geo(data_1, lat='latitude', lon='longitude', color='ciudad_1', size='target')
    fig1.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=0.8
    ))
    layout = html.Div([
        html.Div([
            html.Div([
                dcc.Dropdown(
                    id="drop_table",
                    options=[{"label": x, "value": x} for x in ['nada', 'ciudad', 'universidad_matriculado']],
                    value='nada',
                    clearable=False,
                ),
                dash_table.DataTable(

                    id='table',
                    # columns=[{"name": i, "id": i} for i in data.columns],
                    # data=data.to_dict('records'),
                    columns=[],
                    data=[],
                    fixed_columns={'headers': True, 'data': 1},
                    style_table={'height': '400px', 'overflowY': 'scroll', 'width': '100%'},
                    style_cell_conditional=[
                        {'if': {'column_id': 'empleado_id'},
                         'width': '20%'},
                    ]

               )],
            className='quarter', id='tabla'),
            html.Div([
                dcc.Graph(
                    id='example-graph-2',
                    figure=fig1
                ),
            ], className='quarter')
        ], className='row'),

    html.Div([
        html.Div([
            dcc.Graph(
                id='example-graph-2',
                figure=fig1
            ),
        ])
    ], className='one-row')])
    return layout

def historic_data_layout():
    df = pd.DataFrame({
        "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
        "Amount": [4, 1, 2, 2, 4, 5],
        "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
    })
    fig1 = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")
    data = pd.read_csv('app/data/ds_job.csv')[:50]
    data_plot1 = df_for_plotting1()
    data_plot = df_for_plotting()

    layout = html.Div([
            html.Div([
                dcc.Dropdown(
                    id="dropdown",
                    options=[{"label": x, "value": x} for x in data_plot1.keys()],
                    value=list(data_plot1.keys())[0],
                    clearable=False,
                ),
                dcc.Graph(id="bar-chart"),
            ], className='quarter'),
            html.Div([
                dcc.Dropdown(
                    id="dropdown-circle",
                    options=[{"label": x, "value": x} for x in data_plot.keys()],
                    value=list(data_plot.keys())[0],
                    clearable=False,
                ),
                dcc.Graph(id="bar-chart-circle"),
            ], className='quarter'),
        ], className='row')
    return layout


def model_layout():
    metrics, model_name = load_model_metrics('hipf_db', name=True)
    conf_matrix = metrics['confusion_matrix']
    tp = conf_matrix[0][0]
    fn = conf_matrix[0][1]
    fp = conf_matrix[1][0]
    tn = conf_matrix[1][1]
    conf_matrix = np.asarray(([fn, tp], [tn, fp]))
    conf_matrix = np.asarray(([tp, fn], [fp, tn]))
    print(conf_matrix)
    print(conf_matrix)
    #print(conf_matrix[0][1])
    #conf_matrix = np.asarray(([conf_matrix[0]], [conf_matrix[1]])).T[::-1].T
    print(conf_matrix)
    hover_labels = [['True Positive', 'False Negative'], ['False Positive', 'True Negative']]
    metrics = metrics.drop('confusion_matrix', axis=1).transpose()
    metrics = metrics[0].reset_index()
    metrics.columns = ['metric', 'score']
    values = ['train', 'test']
    fig = ff.create_annotated_heatmap(conf_matrix, x=[1, 0], y=[1, 0],
                                      text=hover_labels, hoverinfo='text',
                                      colorscale=px.colors.sequential.Blues[3:])
    fig.update_layout(title_text='Confusion Matrix', paper_bgcolor="#F7F7F7")

    layout = html.Div([
        html.Div([
            html.H2('Modelo en producción: ' + model_name),
            dash_table.DataTable(

                id='table',
                columns=[{"name": i, "id": i} for i in metrics.columns],
                data=metrics.to_dict('records'),
                # columns=[],
                # data=[],
                fixed_columns={'headers': True, 'data': 1},
                style_table={'height': 'auto', 'width': '100%'},
                style_cell={'textAlign': 'left'}
            ),
                html.Div([
                    dcc.Graph(
                        id='example-graph-conf',
                        figure=fig,
                        style={'height': '300px'}
                    )
                ], className='conf-matrix'),

        ], className='quarter', id='tabla'),
        html.Div([
            html.H2('Comparación frente a otros modelos'),
            dcc.Dropdown(
                id="dropdown-1",
                options=[{"label": x, "value": x} for x in values],
                value=values[0],
                clearable=False,
            ),
            dcc.Graph(id="bar-chart-1"),
        ], className='quarter')

    ], className='row')
    return layout


