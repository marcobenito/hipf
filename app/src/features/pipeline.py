import random
import numpy as np
import pandas as pd


## Función para discretizar la variable experiencia en buckets
def buckets_experiencia(x, limits=[6, 20]):
    try:
        x = int(x)
        for i in limits:
            if x <= i:
                return '<' + str(i)
    except:
        if x == 'nan':
            return np.nan
        return x

# Función que permite unir dos variables discretas
def combine_buckets(x1, x2):
    if str(x1) != 'nan' and str(x2) != 'nan':
        # Cogemos solo la primera palabra de cada variable para que no sea tan larga
        return str(x1).split(' ')[0] + '_' + str(x2).split(' ')[0]
    else:
        return np.nan
    return x1

# Función encargada de llamar a la función combine_buckets de manera correcta para combinar variables
def combine_features(data, col1, col2, new_col_name, f1=None, f2=None):
    """This function combines two features calling the combine_buckets function.
    Args:
    data (DataFrame): whole dataset
    col1 (str): name of the first column to combine
    col2 (str): name of the second column to combine
    new_col_name (str): name for the new variable
    f1 (str) optional: optional function to apply to the first column
    f2 (str) optional: optional function to apply to the second column
    Returns:
    data1 (DataFrame): new dataset with the new variable"""

    data1 = data.copy()
    # Convertimos todos las variables a string
    # Diferenciamos en función de si la entrada es un dataframe o un series
    if isinstance(data1, pd.DataFrame):
        data1['new_' + col1] = data1[col1].apply(lambda x: str(x))
        data1['new_' + col2] = data1[col2].apply(lambda x: str(x))

        # Si hay funciones previas que han de ser aplicadas, lo hacemos
        if f1 != None:
            data1['new_' + col1] = data1[col1].apply(f1)
        if f2 != None:
            data1['new_' + col2] = data1[col2].apply(f2)
        # Creación de nueva feature combinando variables
        data1[new_col_name] = data1.apply(lambda x: combine_buckets(x['new_' + col1], x['new_' + col2]), axis=1)
        # Borramos las variables intermedias
        data1.drop(columns=['new_' + col1, 'new_' + col2], inplace=True)
        return data1

    else:
        data1['new_' + col1] = str(data1[col1])
        data1['new_' + col2] = str(data1[col2])
        if f1 != None:
            data1['new_' + col1] = f1(data1[col1])
        if f2 != None:
            data1['new_' + col2] = f2(data1[col2])
        data1[new_col_name] = combine_buckets(data1['new_' + col1], data1['new_' + col2])
        data1.drop(labels=['new_' + col1, 'new_' + col2], inplace=True)
        return data1

#funcion para calcular cual seria el numero de nulos a rellenar por cada variable en funcion de su peso
def calculo_ponderado(df, nulos):
    df = df.dropna()
    total_datos = df.count()
    datos = df.unique()
    j = 0
    for i in df.unique():
        datos[j] = df[df == i].count()  # sacamos cuantos valores hay de cada tipo en total
        j = j+1
    porcentaje = list(datos)
    u = 0
    for q in datos:
        porcentaje[u] = (datos[u]*100)/total_datos  # calculamos el porcentaje asociado
        u = u+1
    k = 0
    pond = list(datos)
    for valor in porcentaje:
        pond[k] = valor*nulos/100   # calculamos la correspondencia en funcion de la cantidad de nulos
        pond[k] = np.round(pond[k])
        k = k+1
    df_pond = pd.DataFrame(pond)
    df_pond_t = df_pond.T
    df_pond_t.columns = df.unique()  # creamos dataframe
    maximo = df_pond_t.max()
    return df_pond_t, maximo.index[0]


def imputacion_nulos(df, mapeo):   # funcion para imputar nulos
    v = list(mapeo)
#     print(df.shape)
    if df.shape[0] == 1:
        weights = mapeo.values / mapeo.values.sum()
        fill_value = random.choices(range(len(v)), weights=weights, k=1)
        fill_value = v[fill_value]
        df = df.fillna(fill_value)

    else:
        j = 0
    #     print(mapeo.loc['Has relevent experience'])
        for i in v:
    #         print(i)
            t = mapeo.loc[j, i]
            if t > 0:
                df = df.fillna(i, limit= int(t))  # limitamos el numero de nulos a rellenar en funcion de cuantos tiene q añadir de cada tipo
            j+1
        return df