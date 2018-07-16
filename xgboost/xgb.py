import numpy, random
import xgboost as xgb
from matplotlib import pyplot
import lime
import lime.lime_tabular
from xgboost import sklearn

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
    train_data_new = []
    labels = []
    i=0

    # split data into labels and non-labels
    for row in train_data[1:]:
        temp = []
        for x in row[9:]:
            try:
                temp.append(float(x))
            except:
                temp.append(float(-999.0))
        # create label based on relevant columns
        possible_labels = [row[0], row[6], row[7], row[8]]
        labels.append(1 if any(int(label) > 0 for label in possible_labels) else 0)

        train_data_new.append(numpy.asarray(temp))
        i+=1

    train_data = numpy.asarray(train_data_new)

    # shuffle data to make train/test split random
    random.shuffle(train_data)

    labels = numpy.array(labels)

    # split data into train set and test set
    train_split = int(len(train_data) * 0.8)
    X_train, X_test = train_data[:train_split], train_data[train_split:]
    Y_train, Y_test = labels[:train_split], labels[train_split:]

    # create XGBClassifier model
    m = sklearn.XGBClassifier(max_depth=train_args[0], learning_rate=train_args[1], silent=train_args[2],\
                               objective=train_args[3])
    # train model
    m.fit(X=X_train, y=Y_train)

    # predict labels of test set
    preds = m.predict(X_test)

    # calculate error rate of model
    error = sum(1 for i in range(len(preds)) if int(float(preds[i]) > 0.5) != int(Y_test[i])) / float(len(preds))

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
    i=0
    # split data into labels and non-labels
    for row in inquiry_args[1:]:

        temp = []
        for x in row[9:]:
            try:
                temp.append(float(x))
            except:
                temp.append(float(-999.0))

        inquire_data_new.append(numpy.asarray(temp))
        i+=1

    inquire_data = numpy.asarray(inquire_data_new)

    # make predictions and append predictions & probabilities to lists
    probs = model.modelObject.predict_proba(inquire_data)
    preds = model.modelObject.predict(inquire_data)
    prob_list = []
    for i in range(0, len(probs)):
        prob_list.append(float(probs[i][1]))
    pred_list = []
    for i in range(0, len(preds)):
        pred_list.append(preds[i])

    # prediction function
    predict_fn_xgb = lambda x: model.modelObject.predict_proba(x)

    # feature importance explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(inquire_data, feature_names=headers, class_names=[0, 1])

    ret_list = []
    # construct return dicts
    # for i in range (0, len(inquire_data)):
    for i in range(0,5):  # changed to only first 5 dicts to run quickly. revert to line above to return entire inquiry
        exp = explainer.explain_instance(inquire_data[i], predict_fn_xgb, num_features=len(inquire_data[0]))
        exp_list = exp.as_list()

        features = []
        exp_list = sorted(exp_list, key=lambda tup: float(tup[1]), reverse=True)
        for feat in exp_list:
            # add only interpretable features in a readable format
            if feat[1] > 0.0:
                if len(features) < 5:
                    if 'Annual Revenue' in feat[0]:
                        pass
                    elif '<= -999.00' in feat[0]:
                        s = feat[0]
                        s = s.replace(' <= -999.00', '')
                        s = "Missing data for " + s
                        features.append(s)
                    elif '> 50.00' in feat[0]:
                        s = feat[0]
                        s = s.replace(' > 50.00', '')
                        s = "High " + s
                        features.append(s)
                    elif 'has_finance' in feat[0]:
                        if '<= 0.00' in feat[0]:
                            features.append('Job title DOES NOT CONTAIN \'Finance\' or Similar')
                        else:
                            features.append('Job title contains \'Finance\' or Similar')
                    elif 'has_chief' in feat[0]:
                        if '<= 0.00' in feat[0]:
                            features.append('NOT C-suite Job title')
                        else:
                            features.append('C-suite Job title')
                    else:
                        if '<= 0.00' in feat[0]:
                            s = feat[0]
                            s = s.replace(' <= 0.00', '')
                            s = "NOT industry " + s
                            features.append(s)
                        elif '> 0.00' in feat[0]:
                            s = feat[0]
                            s = s.replace(' > 0.00', '')
                            s = "Industry " + s
                            features.append(s)
                        else:
                            features.append(feat[0])
                else:
                    break

        cur_dict = {'ID': contact_ids[i], 'first_name': first_names[i], 'last_name': last_names[i],\
                    'company': company_names[i], 'job_title': job_titles[i], 'probability': prob_list[i],\
                    'features': features}
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
    eta = .3
    silent = 1
    objective = 'binary:logistic'
    num_round = 2
    train_args = [max_depth, eta, silent, objective]

    # train function
    model = train(train_data, train_args)

    # inquiry args
    inquiry_args = csv_to_array('inquire.csv')

    # inquiry function
    print(inquire(model, inquiry_args))

    # plot
    show_plot(model)
