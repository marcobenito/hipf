from app.src.features.pipeline import *

def city(x):
    if x == 0.789:
        return 'Medellin'
    elif x == 0.836:
        return 'Oporto'
    elif x == 0.884:
        return 'Buenos Aires'
    elif x == 0.903:
        return 'Barcelona'
    elif x == 0.913:
        return 'Madrid'
    elif x == 0.915:
        return 'Monterrey'
    elif x == 0.518:
        return 'Mombasa'
    else:
        return 'Otro'

def idh(x):
    if x == 'Medellin':
        return 0.789
    elif x == 'Oporto':
        return 0.836
    elif x == 'Buenos Aires':
        return 0.884
    elif x == 'Barcelona':
        return 0.903
    elif x == 'Madrid':
        return 0.913
    elif x == 'Monterrey':
        return 0.915
    elif x == 'Mombasa':
        return 0.518
    else:
        return 0

def latitude(x):
    if x == 'Medellin':
        return 6.25184
    elif x == 'Oporto':
        return 41.1496
    elif x == 'Buenos Aires':
        return -34.6132
    elif x == 'Barcelona':
        return 41.390205
    elif x == 'Madrid':
        return 40.416775
    elif x == 'Monterrey':
        return 25.67507
    elif x == 'Mombasa':
        return -4.05


def longitude(x):
    if x == 'Medellin':
        return -75.56359
    elif x == 'Oporto':
        return -8.6110
    elif x == 'Buenos Aires':
        return -58.3772
    elif x == 'Barcelona':
        return 2.154007
    elif x == 'Madrid':
        return -3.703790
    elif x == 'Monterrey':
        return -100.31847
    elif x == 'Mombasa':
        return 39.666666667

def city_characteristics():
    data = pd.read_csv('app/data/ds_job.csv')
    new_data = data
    new_data['ciudad_1'] = new_data['indice_desarrollo_ciudad'].apply(city)
    new_data['latitude'] = new_data['ciudad_1'].apply(latitude)
    new_data['longitude'] = new_data['ciudad_1'].apply(longitude)

    new_data = new_data[new_data['ciudad_1'] != 'Otro']
    new_data = new_data[['ciudad_1', 'latitude', 'longitude', 'target']]
    new_data = new_data.groupby(['ciudad_1', 'latitude', 'longitude']).count()
    return new_data
