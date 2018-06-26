# wrapper for the models for convenience


class Model(object):

    def __init__(self, modelObject):
        self.modelObject = modelObject


def train(train_data, train_args):

    return Model('model')


def inquire(model, inquiry_args):

    results = model.modelObject

    return results
