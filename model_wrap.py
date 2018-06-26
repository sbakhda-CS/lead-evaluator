# wrapper for the models for convenience

class my_model(object):
    def __init__(self, modelObject, dtrain):
        self.model = modelObject
        self.dtrain = dtrain


def train(train_data, args):

    modelObject = ''
    dtrain = ''

    model = my_model(modelObject, dtrain)

    return model


def inquire(model, inquiry_set):

    modelObject = model.modelObject
    dtrain = model.dtrain

    results = []

    return results