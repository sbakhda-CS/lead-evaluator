# wrapper for the models for convenience


class Model(object):

    def __init__(self, modelObject):
        self.modelObject = modelObject

    def do_to_text(self, text):
        return text[::-1]


def train(train_data, train_args):

    return Model('')


def inquire(model, inquiry_args):

    results = [model.do_to_text(inquiry_args[0])]

    return results

# create data.csv with column headers in the first row
# change 'mymodel' to your model name and replace in all files
# name your skill and edit your skill.yaml to handle the inputs differently
# create your my_model.py file
# rename docker images everywhere
# test locally
# ./deploy-all.sh
# train from studio
# ./checks.sh [job id]
# inquire in studio