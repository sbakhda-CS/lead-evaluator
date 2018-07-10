import numpy, random
import xgboost as xgb
from matplotlib import pyplot
import lime
import lime.lime_tabular

TEST_FILE = 'TEST_SET_FINAL.csv'

class Model(object):

    def __init__(self, modelObject, error, X_train):
        self.modelObject = modelObject
        self.error = error
        self.X_train = X_train

# convert csv file to array for processing
def csv_to_array(filename):
    a = []
    f = open(filename, 'r')
    for row in f.readlines():
        try:
            a.append([float(x) for x in row.split(',')])
        except:
            a.append([x for x in row.split(',')])
    return a

# train model
def train(train_data, train_args):
    # get headers (column titles)
    headers = train_data[0][9:]

    train_data_new = []
    labels = []

    # split data into labels and non-labels
    for row in train_data[1:]:
        temp = []
        for x in row[9:]:
            try:
                temp.append(float(x))
            except:
                temp.append(float(-999.0))
        for label in row[0]:
            labels.append(label)
        train_data_new.append(numpy.asarray(temp))

    train_data = numpy.asarray(train_data_new)

    # shuffle data to make train/test split random
    random.shuffle(train_data)

    labels = numpy.array(labels)


    # split data into train set and test set
    train_split = int(len(train_data) * 0.8)
    X_train, X_test = train_data[:train_split], train_data[train_split:]
    Y_train, Y_test = labels[:train_split], labels[train_split:]

    # internal data structure to hold train & test sets
    dtrain = xgb.DMatrix(data=X_train, feature_names=headers, missing=-999, label=Y_train)
    dtest = xgb.DMatrix(data=X_test, feature_names=headers, missing=-999, label=Y_test)

    # training parameters
    param = {'max_depth': train_args[1], 'eta': train_args[2], 'silent': train_args[3], 'objective': train_args[4]}

    # specify validations set to watch performance
    watchlist = [(dtest, 'eval'), (dtrain, 'train')]
    num_round = train_args[5]
    m = xgb.train(param, dtrain, num_round, watchlist)

    # predict labels of test set
    preds = m.predict(dtest)

    # calculate error rate of model
    labels = dtest.get_label()
    error = sum(1 for i in range(len(preds)) if int(preds[i] > 0.5) != labels[i]) / float(len(preds))

    final_model = Model(m, error, X_train)
    return final_model


# inquire model
def inquire(model, inquiry_args):
    inquiry_args = numpy.array(inquiry_args)

    # get headers (column titles)
    headers = inquiry_args[0][9:]

    # get ID columns
    contact_ids = inquiry_args[1:, 1]
    first_names = inquiry_args[1:, 2]
    last_names = inquiry_args[1:, 3]
    company_names = inquiry_args[1:, 4]
    job_titles = inquiry_args[1:, 5]

    inquire_data_new = []
    labels = []

    # split data into labels and non-labels
    for row in inquiry_args[1:]:
        temp = []
        for x in row[9:]:
            try:
                temp.append(float(x))
            except:
                temp.append(float(-999.0))
        for label in row[0]:
            labels.append(label)
        inquire_data_new.append(numpy.asarray(temp))
    inquire_data = numpy.asarray(inquire_data_new)

    # internal data structure to hold list of inquire leads
    lead_list = xgb.DMatrix(data=inquire_data, feature_names=headers, missing=-999)

    # make predictions and append to list
    preds = model.modelObject.predict(lead_list)
    pred_list = []
    for i in range(0, len(preds)):
        pred_list.append(preds[i])

    # append prediction of both classes to numpy array (for use in LIME explainer later)
    both_preds = []
    for pred in pred_list:
        temp = []
        temp.append(1-pred)
        temp.append(pred)
        both_preds.append(temp)
    both_preds = numpy.array(both_preds)

    # prediction function
    #predict_fn_xgb = lambda x: model.modelObject.predict_proba(x)

    # feature importance explainer
    #explainer = lime.lime_tabular.LimeTabularExplainer(model.X_train, feature_names=headers, class_names=[0, 1])

    ret_list = []
    # construct return dicts
    for i in range(0, len(inquire_data)):
        cur_dict = {}
        cur_dict['ID'] = contact_ids[i]
        cur_dict['first_name'] = first_names[i]
        cur_dict['last_name'] = last_names[i]
        cur_dict['company'] = company_names[i]
        cur_dict['job_title'] = job_titles[i]
        cur_dict['prob'] = pred_list[i]
        #exp = explainer.explain_instance(model.X_train[i], predict_fn_xgb, num_features=len(train_data[0]))
        #exp_list = exp.as_list()
        # cur_dict['features'] = (x[0] for x in exp_list[:5])
        ret_list.append(cur_dict)

    return ret_list


# plot feature importance
def show_plot(model):
    xgb.plot_importance(model.modelObject)
    pyplot.show()


if __name__ == "__main__":
    # data
    train_data = csv_to_array('train_test.csv')

    # train_args
    max_depth = 6
    eta = .8
    silent = 1
    objective = 'binary:logistic'
    num_round = 2
    train_args = [None, max_depth, eta, silent, objective, num_round]

    # train function
    model = train(train_data, train_args)

    # inquiry args
    inquiry_args = csv_to_array('inquire.csv')

    # inquiry function
    inquire(model, inquiry_args)

    # plot
    show_plot(model)