import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import pandas as pd
import dash
import dash_table
import plotly.express as px
from app.src.utils.utils import df_for_plotting1


def dashboard_layout(dash_app):
    df = pd.DataFrame({
        "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
        "Amount": [4, 1, 2, 2, 4, 5],
        "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
    })
    data = pd.read_csv('app/data/ds_job.csv')[:50]
    data = data[['empleado_id', 'ciudad','genero', 'experiencia', 'universidad_matriculado', 'target']]
    data_plot = df_for_plotting1()

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
                dcc.Dropdown(
                    id="drop_table",
                    options=[{"label": x, "value": x} for x in ['nada', 'ciudad', 'universidad_matriculado']],
                    value='nada',
                    clearable=False,
                ),dash_table.DataTable(

                    id='table',
                    columns=[{"name": i, "id": i} for i in data.columns],
                    data=data.to_dict('records'),
                    #columns=[],
                    #data=[],
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


def dashboard_layout_1(dash_app):
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
    data = data[['empleado_id', 'ciudad', 'genero', 'experiencia', 'universidad_matriculado', 'target']]
    data_plot = df_for_plotting1()

    fig1 = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")
    layout = html.Div([
        html.Div([
            html.Div([
                dcc.Dropdown(
                    id="drop_table",
                    options=[{"label": x, "value": x} for x in ['nada', 'ciudad', 'universidad_matriculado']],
                    value='nada',
                    clearable=False,
                ),dash_table.DataTable(

                    id='table',
                    columns=[{"name": i, "id": i} for i in data.columns],
                    data=data.to_dict('records'),
                    #columns=[],
                    #data=[],
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
    data = data[['empleado_id', 'ciudad', 'genero', 'experiencia', 'universidad_matriculado', 'target']]
    data_plot = df_for_plotting1()

    layout = html.Div([
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
    return layout


def model_layout():
    df = pd.DataFrame({
        "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
        "Amount": [4, 1, 2, 2, 4, 5],
        "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
    })
    fig1 = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")
    values = ['train', 'test']
    layout = html.Div([
        html.Div([
            dcc.Graph(
                id='example-graph-2',
                figure=fig1
            ),
        ], className='quarter'),
        html.Div([
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


