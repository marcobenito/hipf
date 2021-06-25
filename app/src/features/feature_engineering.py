import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# Clase encargada de crear nuevas features, llamando a la función correspondiente
from app.src.features.pipeline import combine_features, calculo_ponderado, imputacion_nulos


class CreateFeatures(TransformerMixin):
    def __init__(self, function, *args, **kwargs):
        self.function = function
        self.args = args
        self.kwargs = kwargs

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.function != combine_features:
            X['new_' + self.kwargs['col1']] = X[self.kwargs['col1']].apply(self.function)
            return X
        return self.function(X, *self.args, **self.kwargs)

    # Clase encargada de eliminar features


class DropFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, features_to_drop):
        self.features_to_drop = features_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if len(X.shape) == 1:
            return X.drop(labels=self.features_to_drop)
        return X.drop(columns=self.features_to_drop)


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction import FeatureHasher


# Clase encargada de realizar un encoding sobre las variables categóricas
class Encoder():
    """This class will execute an encoding scheme to categorical features. Two different schemes can
    be applied: OneHotEncoding and FeatureHasher. The FeatureHaasher scheme allows to reduce the number
    of output variables, in case there are a lot, in order to reduce possible overfitting. Therefore,
    if the number of different values within a feature is higher than *limit*, FeatureHasher scheme
    will be applied. Otherwise, OneHotEncoding scheme will."""

    def __init__(self, limit=8):
        self.limit = limit
        self.ohe = OneHotEncoder()
        self.fh = FeatureHasher(n_features=self.limit, input_type='string')

    def fit(self, X, y=None):
        self.ft_to_hash = []
        for col in X.columns:
            if len(np.unique(X[col])) >= self.limit:
                self.ft_to_hash.append(col)
        X1 = np.asarray(X[self.ft_to_hash])
        X2 = X.drop(columns=self.ft_to_hash)
        self.ohe.fit(X2)
        self.fh.fit(X1)
        return self

    def transform(self, X):

        X1 = np.asarray(X[self.ft_to_hash])
        X2 = X.drop(columns=self.ft_to_hash)
        ohe = pd.DataFrame(self.ohe.transform(X2).toarray())
        fh = pd.DataFrame(self.fh.transform(X1).toarray())
        return pd.concat([ohe, fh], axis=1)

    def fit_transform(self, X, y=None):
        X2 = self.fit(X).transform(X)
        return X2


# Clase encargada de selecccionar las variables adecuadas al inicio del proceso
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names]


# Clase encargada de imputar los nulos con la función SimpleImputer
from sklearn.impute import SimpleImputer


class Imputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy='median'):
        self.strategy = strategy
        self.imputer = SimpleImputer(strategy=self.strategy)

    def fit(self, X, y=None):
        return self.imputer.fit(X)

    def transform(self, X):
        X1 = X.values
        # Diferenciamos entre si la entrada es un dataframe o un series
        if isinstance(X, pd.Series):
            X1 = X1.reshape(1, -1)
            df = pd.DataFrame(self.imputer.transform(X1))
            df.columns = X.index
        else:
            df = pd.DataFrame(self.imputer.transform(X))
            df.columns = X.columns
        return df

    def fit_transform(self, X, y=None):
        df = pd.DataFrame(self.imputer.fit(X).transform(X))
        df.columns = X.columns
        return df


# Clase encargada de imputar los nulos con la función de imputación ponderada
class Imputer1(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        cols = X.columns
        prueba = cols
        j = 0
        df_sin_nul = X.copy()
        for i in cols:
            #             print(X[i].isnull().sum())
            #             print(X[i].value_counts())
            prueba, maximo = calculo_ponderado(X[i], X[i].isnull().sum())
            df_sin_nul[i] = imputacion_nulos(X[i], prueba)
            if df_sin_nul[
                i].isnull().sum() > 0:  # en caso de que queden nulos por rellenar los llenamos con el valor mas alto
                df_sin_nul[i] = df_sin_nul[i].fillna(maximo)
            j = j + 1
        return df_sin_nul


class Imputer2(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.weights = {}
        for col in X.columns:
            df = X[col].dropna()
            self.weights[col] = df.value_counts() / len(df)
        return self

    def transform(self, X):

        if isinstance(X, pd.DataFrame):
            new_data = pd.DataFrame()
            for key in self.weights.keys():
                df = X[key]
                n = df.isna().sum()
                idx = df.loc[pd.isna(df)].index
                fill_values = random.choices(self.weights[key].index, self.weights[key].values, k=n)
                fill_values = pd.Series(fill_values, index=idx)
                df.fillna(fill_values, inplace=True)
                new_data = pd.concat([new_data, df], axis=1)
            new_data.columns = X.columns
            return new_data
        elif isinstance(X, pd.Series):
            for key in self.weights.keys():
                if pd.isna(X[key]):
                    fill_value = random.choices(self.weights[key].index, self.weights[key].values, k=1)
                    X.fillna({key: fill_value[0]}, inplace=True)
            # Trasponemos los datos para que tengan el formato requerido
            return pd.DataFrame(X).transpose()

# Clase encargada de convertir los valores a string
class Stringer(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None):
        #         self.function = function
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        def stringer(x):
            x1 = str(x)
            if x1 == 'nan':
                return np.nan
            else:
                return x1

        if self.cols is None:
            self.cols = X.columns
        for col in self.cols:
            # Diferenciamos entre si la entrada es dataframe o series
            if isinstance(X, pd.DataFrame):
                X[col] = X[col].apply(stringer)
            else:
                X[col] = str(X[col])
        return X


## Función para discretizar la variable índice de desarrollo ciudad en buckets
def buckets_numericos(x, limits = [0.5, 0.6, 0.7, 0.8, 0.9, 1]):
    for i in limits:
        if float(x) < i:
            return '<' + str(i)
    if float(x) > limits[-1]:
        return '>' + str(limits[-1])


## Función para discretizar la variable tamaño de compañía en buckets
def buckets_tam_compania(x):
    tam_c = {'<10': 'S', '10/49': 'S',
             '50-99': 'M', '100-500': 'M',
             '500-999': 'M', '1000-4999': 'L',
             '5000-9999': 'L', '10000+': 'L'}

    if str(x) != 'nan':
        return tam_c[x]
    else:
        return np.nan

## Función para discretizar la variable experiencia en buckets
def buckets_experiencia(x, limits = [10, 20]):
    try:
        x = int(x)
        for i in limits:
            if x <= i:
                return '<' + str(i)
    except:
        if x == 'nan':
            return np.nan
        return x