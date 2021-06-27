from app.src.features.pipeline import *
## Diccionario de ciudades posibles.
## La estructura de la lista es: [idh, latitud, longitud]
cities = {
    'Madrid': [0.92, 40.416775, -3.703790],
    'Barcelona': [0.903, 41.390205, 2.154007],
    'Oporto': [0.836, 41.1496, -8.6110],
    'Buenos Aires': [0.884, -34.6132, -58.3772],
    'Monterrey': [0.915, 25.67507, -100.31847],
    'Medellin': [0.789, 6.25184, -75.56359],
    'Mombasa': [0.601, -4.05, 39.666666667],
    'Bangalore': [0.624, 12.97194, 77.59369],
    'Rabat': [0.682, 34.01325, -6.83255]
}

def city(x):
    """Devuelve el nombre de la ciudad en función del idh"""
    try:
        x = float(x)
    except:
        pass
    for key in cities.keys():
        if x == cities[key][0]:
            return key
    return 'Otro'

def idh(x):
    """Devuelve el idh en función del nombre de la ciudad"""
    try:
        return cities[x][0]
    except:
        return 0

def latitude(x):
    """Devuelve la latitud en función del nombre de la ciudad"""
    try:
        return cities[x][1]
    except:
        return 0


def longitude(x):
    """Devuelve la longitud en función del nombre de la ciudad"""
    try:
        return cities[x][2]
    except:
        return 0