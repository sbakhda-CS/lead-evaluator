import numpy, random
import xgboost as xgb


class Model(object):

    def __init__(self, modelObject, dtrain, error):
        self.modelObject = modelObject
        self.dtrain = dtrain
        self.error = error


def train(train_data, train_args):

    headers = train_data[0][1:]
    train_data = [[float(x) for x in row] for row in train_data[1:]]
    random.shuffle(train_data)

    labels = []
    non_labels = []
    for lead in train_data:
        labels.append(lead[0])
        non_labels.append(lead[1:])

    labels = numpy.array(labels)
    non_labels = numpy.array(non_labels)

    train_split = int(len(non_labels)*0.8)
    final_train, final_test = non_labels[:train_split], non_labels[train_split:]
    final_train_labels, final_test_labels = labels[:train_split], labels[train_split:]

    dtrain = xgb.DMatrix(data=final_train, feature_names=headers, missing=-999, label=final_train_labels)
    dtest = xgb.DMatrix(data=final_test, feature_names=headers, missing=-999, label=final_test_labels)

    # training parameters
    param = {'max_depth': train_args[1], 'eta': train_args[2], 'silent': train_args[3], 'objective': train_args[4]}

    # specify validations set to watch performance
    watchlist = [(dtest, 'eval'), (dtrain, 'train')]
    num_round = train_args[5]
    m = xgb.train(param, dtrain, num_round, watchlist)

    preds = m.predict(dtest)
    labels = dtest.get_label()
    error = sum(1 for i in range(len(preds)) if int(preds[i] > 0.5) != labels[i]) / float(len(preds))
    final_model = Model(m, dtrain, error)
    return final_model


def inquire(model, inquiry_args):

    headers = inquiry_args[0][1:]
    train_data = [[float(x) for x in row] for row in inquiry_args[1:]]

    lead_list = xgb.DMatrix(data=numpy.array([x[1:] for x in train_data]), feature_names=headers, missing=-999)

    # make predictions
    preds = model.modelObject.predict(lead_list)

    pred_list = []
    for i in range(0, len(preds)):
        pred_list.append(preds[i])

    results = pred_list

    return results
