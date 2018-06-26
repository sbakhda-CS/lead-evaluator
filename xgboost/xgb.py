import csv
import xgboost as xgb
from matplotlib import pyplot

TRAIN_FILE = 'TRAIN_SET_FINAL.csv'
TEST_FILE = 'TEST_SET_FINAL.csv'

class Model(object):

    def __init__(self, modelObject, dtrain):
        self.modelObject = modelObject
        self.dtrain = dtrain

def train(dtrain, dtest, max_depth=6, eta=.8, silent=1, objective='binary:logistic', num_round=2):

    # training parameters
    param = {'max_depth': max_depth, 'eta': eta, 'silent': silent, 'objective': objective}

    # specify validations set to watch performance
    watchlist = [(dtest, 'eval'), (dtrain, 'train')]
    model = xgb.train(param, dtrain, num_round, watchlist)
    #model = model(modelObject, dtrain)
    model.save_model('0001.model')
    # dump model
    model.dump_model('dump.raw.txt')
    # dump model with feature map
    model.dump_model('dump.nice.txt')

    return Model(model, dtrain)

def inquire(model, dtest):
    # make predictions
    preds = model.modelObject.predict(dtest)
    labels = dtest.get_label()
    # print error rate of test
    #print('error=%f' % (sum(1 for i in range(len(preds)) if int(preds[i] > 0.5) != labels[i]) / float(len(preds))))

    pred_list = []
    for i in range(0,len(preds)):
        pred_list.append(preds[i])

    results = pred_list
    print(results)
    return results

# plot feature importance
def show_plot(model):
    xgb.plot_importance(model.modelObject)
    pyplot.show()

if __name__ == "__main__":

    # get names of columns
    with open(TRAIN_FILE, newline='') as f:
        reader = csv.reader(f)
        col_names = next(reader)
    col_names = col_names[1:]

    # make data matrix for train data
    dtrain = xgb.DMatrix(TRAIN_FILE + '?format=csv&label_column=0', feature_names=col_names, missing=-999)
    # make data matrix for test data
    dtest = xgb.DMatrix(TEST_FILE + '?format=csv&label_column=0', feature_names=col_names, missing=-999)
    model = train(dtrain, dtest)

    inquire(model, dtest)
    show_plot(model)