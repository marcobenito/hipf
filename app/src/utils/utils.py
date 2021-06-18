from cloudant.client import Cloudant
import ibm_boto3
from ibm_botocore.client import Config
from ibm_botocore.client import ClientError
import pickle
from io import BytesIO

import time
import torch
import random
import functools
import numpy as np
from typing import Any, Callable, TypeVar, cast


def random_seed(seed_value: int) -> None:
    """
    Random Seeds Numpy, Random and Torch libraries

    Args:
        seed_value (int): Number for seeding
    """
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu vars
    random.seed(seed_value)  # Python
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False


F = TypeVar('F', bound=Callable[..., Any])


def timer(func: F) -> F:
    """ Print the runtime of the decorated function """
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        _ = time.perf_counter() - start_time
        hours, _ = divmod(_, 3600)
        minutes, seconds = divmod(_, 60)

        print(f'Execution time of function {func.__name__!r}: {hours:.0f} hrs {minutes:.0f} mins {seconds:.3f} secs')
        return value
    return cast(F, wrapper_timer)





class DocumentDB:
    """
        Clase para gestionar la base de datos documental IBM Cloudant
    """

    def __init__(self, username, api_key):
        """
            Constructor de la conexión a IBM cloudant

            Args:
               username (str): usuario.
               apikey (str): API key.
        """
        self.connection = Cloudant.iam(username, api_key, connect=True)
        self.connection.connect()

    def get_database(self, db_name):
        """
            Función para obtener la base de datos elegida.

            Args:
               db_name (str):  Nombre de la base de datos.

            Returns:
               Database. Conexión a la base de datos elegida.
        """
        return self.connection[db_name]

    def database_exists(self, db_name):
        """
            Función para comprobar si existe la base de datos.

            Args:
               db_name (str):  Nombre de la base de datos.

            Returns:
               boolean. Existencia o no de la base de datos.
        """
        return self.get_database(db_name).exists()

    def create_document(self, db, document_dict):
        """
            Función para crear un documento en la base de datos

            Args:
               db (str):  Conexión a una base de datos.
               document_dict (dict):  Documento a insertar.
        """
        db.create_document(document_dict)


class IBMCOS:
    """
        Clase para gestionar el repositorio de objetos IBM COS
    """

    def __init__(self, ibm_api_key_id, ibm_service_instance_id, ibm_auth_endpoint, endpoint_url):
        """
            Constructor de la conexión a IBM COS

            Args:
               ibm_api_key_id (str): API key.
               ibm_service_instance_id (str): Service Instance ID.
               ibm_auth_endpoint (str): Auth Endpoint.
               endpoint_url (str): Endpoint URL.
        """


        self.connection = ibm_boto3.resource("s3",
                                             ibm_api_key_id=ibm_api_key_id,
                                             ibm_service_instance_id=ibm_service_instance_id,
                                             ibm_auth_endpoint=ibm_auth_endpoint,
                                             config=Config(signature_version="oauth"),
                                             endpoint_url=endpoint_url)

    def save_object_in_cos(self, obj, name, timestamp, bucket_name='models-hifp'):
        """
            Función para guardar objeto en IBM COS.

            Args:
               obj:  Objeto a guardar.
               name (str):  Nombre del objeto a guardar.
               timestamp (float): Segundos transcurridos.

            Kwargs:
                bucket_name (str): depósito de COS elegido.
        """

        # objeto serializado
        pickle_byte_obj = pickle.dumps(obj)
        # nombre del objeto en COS
        pkl_key = name + "_" + str(int(timestamp)) + ".pkl"

        try:
            # guardado del objeto en COS
            self.connection.Object(bucket_name, pkl_key).put(
                Body=pickle_byte_obj
            )
        except ClientError as be:
            print("CLIENT ERROR: {0}\n".format(be))
        except Exception as e:
            print("Unable to create object: {0}".format(e))

    def get_object_in_cos(self, key, bucket_name='models-hifp'):
        """
            Función para obtener un objeto de IBM COS.

            Args:
               key (str):  Nombre del objeto a obtener de COS.

            Kwargs:
                bucket_name (str): depósito de COS elegido.

            Returns:
               obj. Objeto descargado.
        """

        # conexión de E/S de bytes
        with BytesIO() as data:
            # descarga del objeto desde COS
            self.connection.Bucket(bucket_name).download_fileobj(key, data)
            data.seek(0)
            # des-serialización del objeto descargado
            obj = pickle.load(data)
        return obj
