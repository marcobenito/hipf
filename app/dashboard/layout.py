import dash_html_components as html
import dash_core_components as dcc
#import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import dash
import dash_table
import plotly.express as px
import plotly.figure_factory as ff
from app.src.utils.utils import df_for_plotting1, df_for_plotting, df_for_map, plot_nlu
from app.src.models.train_model import load_model_metrics
from app.src.data.conectBBDD import select_table_pred



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

    layout = html.Div([
        html.Div([
            dcc.RadioItems(
                id='data_view',
                options=[
                    {'label': 'Vista genérica', 'value': 'general'},
                    {'label': 'Vista mapa', 'value': 'map'}
                ],
                value='general',
                labelStyle={'display': 'inline-block', 'marginLeft': '7.5%'}
            ),

            html.Div(id='container-1')

        ]),

    html.Div([
        html.Div([
            html.Div([
                html.H2('Buscar por ID de empleado  ', style={'display': 'inline-block'}),
                dcc.Input(id='input-on-submit', type='text'),
                html.Button('Buscar', id='buscar-id', n_clicks=0)
            ], style={'textAlign': 'left'}),
            dcc.Graph(
                id='bar-graph-nlu',
                #figure=fig2
            ),
        ])
    ], className='one-row')])
    return layout

def historic_data_layout():

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




def layout_general():

    data = select_table_pred()
    data_table = data[['empleado_id', 'ciudad', 'target']]
    data_chart = data.drop(columns=['target', 'empleado_id', 'horas_formacion',
                                    'tipo_compania', 'tamano_compania',
                                    'indice_desarrollo_ciudad'])

    layout = html.Div([
        html.Div([
            # dcc.Input(id='input-state', type='text'),
            # html.Button(id='submit-button-state', n_clicks=0, children='Buscar'),
            dash_table.DataTable(
                id='table-pred',
                columns=[
                    {"name": 'ID empleado', "id": 'empleado_id', 'type': 'numeric'},
                    {"name": 'Ciudad', "id": 'ciudad', 'type': 'text'},
                    {"name": 'Predicción', "id": 'target', 'type': 'numeric'},
                         ],
                filter_action='native',
                data=data_table.to_dict('records'),
                # columns=[],

                fixed_columns={'headers': True, 'data': 1},
                fixed_rows={'headers': True},
                style_cell={'textAlign': 'left'},
                style_table={'height': '400px', 'width': '100%'},
                style_data={
                    'width': '{}%'.format(100. / len(data_table.columns)),
                    'textOverflow': 'hidden'
                },
                style_header={
                    'backgroundColor': '#3d3b4e',
                    'color': '#f2f2f2'
                }
            )
        ], className='quarter', id='tabla'),
    html.Div([
        dcc.Dropdown(
            id="dropdown-3",
            options=[{"label": x, "value": x} for x in data_chart.columns],
            value=data_chart.columns[0],
            clearable=False,
        ),
        dcc.Graph(
            id='bar-chart-3',
        ),
    ], className='quarter')
        ], className='row')

    return layout


def layout_map():
    data = select_table_pred()
    data = df_for_map(data)
    fig1 = px.scatter_geo(data, lat='latitude', lon='longitude', color='ciudad', size='N')
    fig1.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=0.8
    ))
    layout = html.Div([
        html.Div([
            dcc.Graph(
                id='example-graph-2',
                figure=fig1
            ),
        ])
    ], className='one-row-1')

    return layout




