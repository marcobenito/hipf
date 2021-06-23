import os
import pickle

import joblib
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import MinMaxScaler

from ..data.make_dataset import make_dataset
from ..data.make_dataset import split_dataset
from ..data.make_dataset import make_RandomOverSampler
from ..data.make_dataset import get_raw_pikle_from_local
from sklearn.pipeline import Pipeline, FeatureUnion
from app import ROOT_DIR, cos, client
from sklearn.ensemble import RandomForestClassifier
from cloudant.query import Query
from xgboost import XGBRFClassifier
import time

from ..evaluation.evaluate_model import evaluate_model
from ..features.feature_engineering import DataFrameSelector, CreateFeatures, DropFeatures, Stringer, Imputer, Encoder
from ..features.pipeline import buckets_experiencia, combine_features


def training_pipeline(path, model_info_db_name='hipf_db'):
    """
        Función para gestionar el pipeline completo de entrenamiento
        del modelo.

        Args:
            path (str):  Ruta hacia los datos.

        Kwargs:
            model_info_db_name (str):  base de datos a usar para almacenar
            la info del modelo.
    """

    # Carga de la configuración de entrenamiento
    model_config = load_model_config(model_info_db_name)['model_config']

    print('*********model_config---->', model_config)

    # timestamp usado para versionar el modelo y los objetos
    ts = time.time()

    # carga y transformación de los datos de train y test
    df = make_dataset(path, ts)

    X_train, X_test,y_train,y_test = split_dataset(df)


    X_train_res, y_train1 = make_RandomOverSampler(X_train, y_train)


    cat_cols = [i for i in df if df[str(i)].dtype == 'O']
    num_cols = [i for i in df if df[str(i)].dtype != 'O']
    num_cols.remove('target')

    # pasos del pipeline categórico
    cat_steps = [
        ('selector', DataFrameSelector(cat_cols)),
        ('imputer', Imputer(strategy='most_frequent')),
        # Creamos las features que necesitemos
        ('creator1', CreateFeatures(combine_features, col1='experiencia', col2='experiencia_relevante',
                                    new_col_name='exp_exp-rel', f1=buckets_experiencia)),
        ('dropper', DropFeatures(['tipo_compania', 'ciudad', 'educacion'])),
        ('stringer', Stringer()),
        ('encoder', Encoder(limit=20))
    ]

    # Pasos del pipeline numérico
    num_steps = [
        ('selector', DataFrameSelector(num_cols)),
        ('dropper', DropFeatures(['empleado_id'])),
        ('imputer', Imputer(strategy='median')),
        ('scaler', MinMaxScaler())
    ]

    num_pipeline = Pipeline(num_steps)
    cat_pipeline = Pipeline(cat_steps)

    # Concatenación de pipelines
    full_pipeline = FeatureUnion([
        ('numeric_pipeline', num_pipeline),
        ('categorical_pipeline', cat_pipeline)
    ])




    # Aplicamos el pipeline de transformación
    pipe = full_pipeline.fit(X_train_res)
    X_train1 = pipe.transform(X_train_res)
    X_test1 = pipe.transform(X_test)
    path = os.path.join('checkpoint/pipeline.pkl')
    with open(path, 'wb') as f:
        pickle.dump(pipe, f)

    print('------>Cargamos el Pickle con los mejores parametros del grid search ')

    pickle_path = os.path.join('checkpoint/parameters.pkl')

    params = get_raw_pikle_from_local(pickle_path)

    print('------>Los mejores parametros',params)

##/*****************Probamos a meter los el modelo en el pipeine *****************/




    print('------>Fit del modelo  XGBRFClassifier')
    clf = XGBRFClassifier(**params)
    clf.fit(X_train1, y_train1)

    print('Train score {}'.format(clf.score(X=X_train1, y=y_train1)))
    print('Test score {}'.format(clf.score(X=X_test1, y=y_test)))

    print('------> Saving the model {} object on the cloud'.format('model_' + str(int(ts))))
    save_model(clf, 'model', ts)

    #y_pred = clf.predict(X_test1)
    # Evaluación del modelo y recolección de información relevante
    print('---> Evaluating the model')
    metrics_dict = evaluate_model(clf, X_test1, y_test, ts, model_config['model_name'])

    # Guardado de la info del modelo en BBDD documental
    print('------> Saving the model information on the cloud')
    info_saved_check = save_model_info(model_info_db_name, metrics_dict)

    # Check de guardado de info del modelo
    if info_saved_check:
        print('------> Model info saved SUCCESSFULLY!!')
    else:
        if info_saved_check:
            print('------> ERROR saving the model info!!')

    # selección del mejor modelo para producción
    print('---> Putting best model in production')
    put_best_model_in_production(metrics_dict, model_info_db_name)






def save_model(obj, name, timestamp, bucket_name='models-hifp'):
    """
        Función para guardar el modelo en IBM COS

        Args:
            obj (sklearn-object): Objeto de modelo entrenado.
            name (str):  Nombre de objeto a usar en el guardado.
            timestamp (float):  Representación temporal en segundos.

        Kwargs:
            bucket_name (str):  depósito de IBM COS a usar.
    """

    print('---> Función Save_object_in_cos ',obj)
    cos.save_object_in_cos(obj, name, timestamp)


def save_model_info(db_name, metrics_dict):
    """
        Función para guardar la info del modelo en IBM Cloudant

        Args:
            db_name (str):  Nombre de la base de datos.
            metrics_dict (dict):  Info del modelo.

        Returns:
            boolean. Comprobación de si el documento se ha creado.
    """
    db = client.get_database(db_name)
    client.create_document(db, metrics_dict)

    return metrics_dict['_id'] in db


def put_best_model_in_production(model_metrics, db_name):
    """
        Función para poner el mejor modelo en producción.

        Args:
            model_metrics (dict):  Info del modelo.
            db_name (str):  Nombre de la base de datos.
    """

    # conexión a la base de datos elegida
    db = client.get_database(db_name)
    # consulta para traer el documento con la info del modelo en producción
    query = Query(db, selector={'status': {'$eq': 'in_production'}})
    res = query()['docs']
    #  id del modelo en producción
    best_model_id = model_metrics['_id']

    # en caso de que SÍ haya un modelo en producción
    if len(res) != 0:
        # se realiza una comparación entre el modelo entrenado y el modelo en producción
        best_model_id, worse_model_id = get_best_model(model_metrics, res[0])
        # se marca el peor modelo (entre ambos) como "NO en producción"
        worse_model_doc = db[worse_model_id]
        worse_model_doc['status'] = 'none'
        # se actualiza el marcado en la BDD
        worse_model_doc.save()
    else:
        # primer modelo entrenado va a automáticamente a producción
        print('------> FIRST model going in production')

    # se marca el mejor modelo como "SÍ en producción"
    best_model_doc = db[best_model_id]
    best_model_doc['status'] = 'in_production'
    # se actualiza el marcado en la BDD
    best_model_doc.save()


def get_best_model(model_metrics1, model_metrics2):
    """
        Función para comparar modelos.

        Args:
            model_metrics1 (dict):  Info del primer modelo.
            model_metrics2 (str):  Info del segundo modelo.

        Returns:
            str, str. Ids del mejor y peor modelo en la comparación.
    """

    # comparación de modelos usando la métrica AUC score.
    auc1 = model_metrics1['model_metrics']['roc_auc_score']
    auc2 = model_metrics2['model_metrics']['roc_auc_score']
    print('------> Model comparison:')
    print('---------> TRAINED model {} with AUC score: {}'.format(model_metrics1['_id'], str(round(auc1, 3))))
    print('---------> CURRENT model in PROD {} with AUC score: {}'.format(model_metrics2['_id'], str(round(auc2, 3))))

    # el orden de la salida debe ser (mejor modelo, peor modelo)
    if auc1 >= auc2:
        print('------> TRAINED model going in production')
        return model_metrics1['_id'], model_metrics2['_id']
    else:
        print('------> NO CHANGE of model in production')
        return model_metrics2['_id'], model_metrics1['_id']


def load_model_config(db_name):
    """
        Función para cargar la info del modelo desde IBM Cloudant.

        Args:
            db_name (str):  Nombre de la base de datos.

        Returns:
            dict. Documento con la configuración del modelo.
    """
    db = client.get_database(db_name)
    query = Query(db, selector={'_id': {'$eq': 'model_config'}})
    return query()['docs'][0]
