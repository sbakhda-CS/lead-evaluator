from cortex_client.webserver import webserver_app
from cortex_client import ModelRouter
from cortex_client import ModelRunner

# from model import CortexModel

from cortex_client.modelprocess import ModelProcess
import pickle
import my_model


class CortexModel(ModelProcess):

    name = 'CortexModel'

    @staticmethod
    def train(request, cortex_model, datasets_client, model_client) -> None:

        data = CortexModel._fetch_training_data(request, datasets_client)

        model = my_model.train(data, request['train_args'])

        # save
        serialized_model = CortexModel._serialize("serialized_model", model)
        model_client.save_state(cortex_model, "serialized_model", serialized_model)

    @staticmethod
    def inquire(request, cortex_model, model_client):

        ser_model = model_client.load_state(cortex_model, "serialized_model").read()
        model = CortexModel._deserialize('serialized_model', ser_model)

        result = my_model.inquire(model, request['inquiry_args'])

        return {"payload": {"result": result}}

    @staticmethod
    def _fetch_training_data(request, datasets_client):

        data = datasets_client.get_stream(request['dataset'])
        data = [l.split(',') for l in data.read().decode('utf-8').split('\n')[0:-1]]

        return data

    @staticmethod
    def _serialize(key: str, datum: object) -> bytes:
        serializers = {'serialized_model': pickle.dumps}
        return serializers[key](datum)

    @staticmethod
    def _deserialize(key: str, datum: bytes) -> object:
        deserializers = {'serialized_model': pickle.loads}
        return deserializers[key](datum)


webserver_app.modelrunner = ModelRunner(CortexModel())

if __name__ == '__main__':

    ModelRouter.main(CortexModel())
