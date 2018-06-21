import csv
import xgboost as xgb
from matplotlib import pyplot

TRAIN_FILE = 'TRAIN_SET_FINAL.csv'
TEST_FILE = 'TEST_SET_FINAL.csv'

# train model
def train(train_file, test_file):
    # get names of columns
    with open(train_file, newline='') as f:
        reader = csv.reader(f)
        col_names = next(reader)
    col_names = col_names[1:]
    # make data matrix for train data
    dtrain = xgb.DMatrix(train_file + '?format=csv&label_column=0', feature_names=col_names, missing=-999)
    # make data matrix for test data
    dtest = xgb.DMatrix(test_file + '?format=csv&label_column=0', feature_names=col_names, missing=-999)
    # training parameters
    param = {'max_depth': 6, 'eta': .8, 'silent': 1, 'objective': 'binary:logistic'}

    # specify validations set to watch performance
    watchlist = [(dtest, 'eval'), (dtrain, 'train')]
    num_round = 2
    model = xgb.train(param, dtrain, num_round, watchlist)

    model.save_model('0001.model')
    # dump model
    model.dump_model('dump.raw.txt')
    # dump model with feature map
    model.dump_model('dump.nice.txt')

    return (model, dtest)

# run test set against model
def test(model, dtest):
    # make predictions
    preds = model.predict(dtest)
    labels = dtest.get_label()
    # print error rate of test
    print('error=%f' % (sum(1 for i in range(len(preds)) if int(preds[i] > 0.5) != labels[i]) / float(len(preds))))

# plot feature importance
def show_plot(model):
    xgb.plot_importance(model)
    pyplot.show()

if __name__ == "__main__":
    model, dtest = train(TRAIN_FILE, TEST_FILE)
    test(model, dtest)
    #show_plot(model)
