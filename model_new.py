from collections import namedtuple
import codecs
import io
import json
import pickle


from cortex_client.modelprocess import ModelProcess


class RealestateRF(ModelProcess):

    name = 'RealestateRF'

    @staticmethod
    def train(request, cortex_model, datasets_client, model_client) -> None:

        # fetching data
        data = RealestateRF._fetch_training_data(request, datasets_client)

        # get trained model
        trained_model = 'asdkfbsajkdfk'

        # testing the data and logging the score
        validation = '0.9'
        model_client.log_event(cortex_model, 'train.validation', validation)

        # save MyModel class (this)
        serialized_trained_model = RealestateRF._serialize("serialized_model", trained_model)
        model_client.save_state(cortex_model, "serialized_model", serialized_trained_model)


    @staticmethod
    def inquire(request, cortex_model, model_client):

        ser_model = model_client.load_state(cortex_model, "serialized_model").read()
        trained_model = RealestateRF._deserialize('serialized_model', ser_model)

        # args = list(request['args'])[0][0]

        if trained_model == 'asdkfbsajkdfk':
            result = 5
        else:
            result = 0

        return {"payload": [['result'],[result]]}


    @staticmethod
    def _fetch_training_data(request, datasets_client):
        data = datasets_client.get_stream(request['dataset'])
        data = [l.split('\t') for l in data.read().decode('utf-8').split('\n')[0:-1]]

        return data

    @staticmethod
    def _serialize(key: str, datum: object) -> bytes:
        serializers = {'serialized_model': pickle.dumps}
        return serializers[key](datum)

    @staticmethod
    def _deserialize(key: str, datum: bytes) -> object:
        deserializers = {'serialized_model': pickle.loads}
        return deserializers[key](datum)
