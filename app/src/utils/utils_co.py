from app.src.features.pipeline import *
## Diccionario de ciudades posibles.
## La estructura de la lista es: [idh, latitud, longitud]
cities = {
    'Madrid': [0.92, 40.416775, -3.703790],
    'Barcelona': [0.903, 41.390205, 2.154007],
    'Oporto': [0.836, 41.1496, -8.6110],
    'Buenos Aires': [0.884, -34.6132, -58.3772],
    'Monterrey': [0.915, 25.67507, -100.31847],
    'Medellín': [0.789, 6.25184, -75.56359],
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


# def city(x):
#     try:
#         x = float(x)
#     except:
#         pass
#     if x == 0.789:
#         return 'Medellin'
#     elif x == 0.836:
#         return 'Oporto'
#     elif x == 0.884:
#         return 'Buenos Aires'
#     elif x == 0.903:
#         return 'Barcelona'
#     elif x == 0.913:
#         return 'Madrid'
#     elif x == 0.915:
#         return 'Monterrey'
#     elif x == 0.518:
#         return 'Mombasa'
#     elif x == 0.624:
#         return 'Bangalore'
#     else:
#         return 'Otro'



# def idh(x):
#     if x == 'Medellin':
#         return 0.789
#     elif x == 'Oporto':
#         return 0.836
#     elif x == 'Buenos Aires':
#         return 0.884
#     elif x == 'Barcelona':
#         return 0.903
#     elif x == 'Madrid':
#         return 0.913
#     elif x == 'Monterrey':
#         return 0.915
#     elif x == 'Mombasa':
#         return 0.518
#     elif x == 'Bangalore':
#         return 0.624
#     else:
#         return 0


# def latitude(x):
#     if x == 'Medellin':
#         return 6.25184
#     elif x == 'Oporto':
#         return 41.1496
#     elif x == 'Buenos Aires':
#         return -34.6132
#     elif x == 'Barcelona':
#         return 41.390205
#     elif x == 'Madrid':
#         return 40.416775
#     elif x == 'Monterrey':
#         return 25.67507
#     elif x == 'Mombasa':
#         return -4.05
#     elif x == 'Bangalore':
#         return 12.97194
#     else:
#         return 0
#
#
# def longitude(x):
#     if x == 'Medellin':
#         return -75.56359
#     elif x == 'Oporto':
#         return -8.6110
#     elif x == 'Buenos Aires':
#         return -58.3772
#     elif x == 'Barcelona':
#         return 2.154007
#     elif x == 'Madrid':
#         return -3.703790
#     elif x == 'Monterrey':
#         return -100.31847
#     elif x == 'Mombasa':
#         return 39.666666667
#     elif x == 'Bangalore':
#         return 77.59369
#     else:
#         return 0
