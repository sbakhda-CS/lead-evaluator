from collections import namedtuple
import codecs
import io
import json
import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy
import sklearn.datasets

from cortex_client.modelprocess import ModelProcess


class Mode(object):
    def __init__(self):
        self.msg = 'im modelasdgkjnasgfkjlwnalfk'

class RealestateRF(ModelProcess):

    name = 'RealestateRF'

    @staticmethod
    def train(request, cortex_model, datasets_client, model_client) -> None:
        """
        `train` is responsible for :
            1. pulling the training data
            2. training
            3. storing / saving the 'learned params' / 'trained model'
        """
        data = RealestateRF._fetch_training_data(request, datasets_client)
        X = data.data
        y = data.target

        a = Mode()
        # save
        b = RealestateRF._serialize("serialized_model", a)
        model_client.save_state(cortex_model, "serialized_model", b)

    @staticmethod
    def inquire(request, cortex_model, model_client):
        """`inquire` is responsible for:
            1. loading the 'trained model'
            2. predicting
        """
        result = ''
        try:
            ser_model = model_client.load_state(cortex_model, "serialized_model").read()
            a = RealestateRF._deserialize('serialized_model', ser_model)
            result  = {"result": a.msg}
        except Exception:
            result = {"result": {"error": "exception while fetching latest model"}}

        return {"payload": result}

    ## Private ##

    @staticmethod
    def _fetch_training_data(request, datasets_client):
        if request['dataset'] == 'scikitlearn':
            return sklearn.datasets.load_boston()
        else:
            ## TODO: update cortex-datasets to use v3 of catalog api
            #data = datasets_client.get_dataframe(request['dataset'])
            #data2 = numpy.array(data['values']).astype(numpy.float)
            data = datasets_client.get_stream(request['dataset'])
            data = [l.split('\t') for l in data.read().decode('utf-8').split('\n')[0:-1]]

            data2 = numpy.array(data).astype(numpy.float)
            X = data2[:,0:-1]
            y = data2[:,-1]
            return namedtuple("Data", ("data", "target"))(X, y)

    @staticmethod
    def _serialize(key: str, datum: object) -> bytes:
        serializers = {'serialized_model': pickle.dumps}
        return serializers[key](datum)

    @staticmethod
    def _deserialize(key: str, datum: bytes) -> object:
        deserializers = {'serialized_model': pickle.loads}
        return deserializers[key](datum)