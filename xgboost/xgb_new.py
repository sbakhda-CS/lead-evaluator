import numpy, random
import xgboost as xgb
from matplotlib import pyplot
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
    i = 0

    # split data into labels and non-labels
    for row in train_data[1:]:
        temp = []
        for x in row[9:]:
            try:
                temp.append(float(x))
            except:
                temp.append(float(-999.0))
        # create label based on relevant columns
        possible_titles = [row[11], row[12]]
        good_titles = ["VP", "CHIEF", "DIRECTOR", "ANALYST", "CEO", "CIO", "CFO", "CTO", "FINANCE", "FINANCIAL",\
                       "PRESIDENT", "INNOVATION", "SENIOR", "OFFICER", "INNOVATIVE", "ANALYSIS"]
        good_company_names = ["HOSPITAL", "BANK", "FINANCE", "FINANCIAL", "HEALTH", "HEALTHCARE", "RETAIL", "APPAREL",\
                              "INVESTMENT", "INVESTING"]
        possible_industries = [row[15], row[17], row[19], row[23], row[42], row[55], row[69], row[74], row[116], row[117]]
        labels.append(1 if (any(int(label) > 0 for label in possible_titles)\
                            or any(t in str(row[5]).lower() for t in good_titles))\
                           or (any(int(label) > 0 for label in possible_industries)\
                                or any(c in str(row[4].lower()) for c in good_company_names)) else 0)

        train_data_new.append(numpy.asarray(temp))
        i += 1

    train_data = numpy.asarray(train_data_new)

    # shuffle data to make train/test split random
    random.shuffle(train_data)

    labels = numpy.array(labels)

    # split data into train set and test set
    train_split = int(len(train_data) * 0.8)
    X_train, X_test = train_data[:train_split], train_data[train_split:]
    Y_train, Y_test = labels[:train_split], labels[train_split:]

    # create XGBClassifier model
    m = sklearn.XGBClassifier(max_depth=train_args[0], learning_rate=train_args[1], silent=train_args[2], \
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
    hubspot_scores = inquiry_args[1:, 10]

    inquire_data_new = []
    i = 0
    # split data into labels and non-labels
    for row in inquiry_args[1:]:

        temp = []
        for x in row[9:]:
            try:
                temp.append(float(x))
            except:
                temp.append(float(-999.0))

        inquire_data_new.append(numpy.asarray(temp))
        i += 1

    inquire_data = numpy.asarray(inquire_data_new)

    # make predictions and append predictions & probabilities to lists
    probs = model.modelObject.predict_proba(inquire_data)
    preds = model.modelObject.predict(inquire_data)

    positives = []
    prob_list = []
    for i in range(0, len(probs)):
        prob_list.append(float(probs[i][1]))
        if float(probs[i][1]) >= 0.5:
            positives.append(i)
    pred_list = []
    for i in range(0, len(preds)):
        pred_list.append(preds[i])

    healthcare_company_names = ["HOSPITAL", "HEALTH", "HEALTHCARE"]
    financial_company_names = ["BANK", "FINANCE", "FINANCIAL", "INVESTMENT", "INVESTING", "BANKING"]
    retail_company_names = ["RETAIL", "APPAREL"]

    decision_titles = ["VP", "CHIEF", "CEO", "CIO", "CFO", "CTO", "PRESIDENT", "SENIOR", "OFFICER", "DIRECTOR"]
    financial_titles = ["CFO", "FINANCE", "FINANCIAL"]
    innovation_titles = ["INNOVATION", "INNOVATIVE"]
    analyst_titles = ["ANALYST", "ANALYSIS"]

    ret_list = []
    # construct return dicts
    # for i in range (0, len(inquire_data)):
    for i in range(0,10):  # changed to only first 5 dicts to run quickly. revert to line above to return entire inquiry
        features = []

        if any(c in str(company_names[i]).upper() for c in healthcare_company_names):
            features.append("Industry: healthcare")
        if any(c in str(company_names[i]).upper() for c in financial_company_names):
            features.append("Industry: financial services")
        if any(c in str(company_names[i]).upper() for c in retail_company_names):
            features.append("Industry: retail")
        if any(t in str(job_titles[i]).upper() for t in decision_titles):
            features.append("Job title: decision-making")
        if any(t in str(job_titles[i]).upper() for t in financial_titles):
            features.append("Job title: finance")
        if any(t in str(job_titles[i]).upper() for t in innovation_titles):
            features.append("Job title: innovation")
        if any(t in str(job_titles[i]).upper() for t in analyst_titles):
            features.append("Job title: analyst")
        elif prob_list[i] > 0.5 and len(features) == 0:
            features.append("Industry: healthcare, financial services, or retail")

        cur_dict = {'ID': contact_ids[i], 'first_name': first_names[i], 'last_name': last_names[i], \
                    'company': company_names[i], 'job_title': job_titles[i], 'probability': prob_list[i], \
                    'features': features}
        ret_list.append(cur_dict)
        print(cur_dict)

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
