import time

import joblib
import numpy as np
import pandas as pd
from imblearn.ensemble import RUSBoostClassifier
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

import reader

'''
    fixed random seed
'''
np.random.seed(0)


class Result:
    def __init__(self, firms, data):
        self.firms = firms
        self.data = data


def learning_model(year, class_weight):
    iters = 300
    gap = 2
    year_test = year

    data_test = reader.ordinary_data_reader('uscecchini28.csv', year_test, year_test)
    x_test = data_test.features
    y_test = data_test.labels
    test = np.c_[data_test.years, data_test.firms]

    '''
        an if-else is used to judge whether the class_weight is None to prevent Exception from string concatenation
        
        a try-except for RusBoost with DecisionTreeClassifier using custom class_weight
        
        if we can find the right model trained last time on disk, we can directly use that model to predict
        the result without training twice
        otherwise, we have to train that model and save it on disk
        
    '''
    # if class_weight is not None:
    # we use current_model_name to find/save the trained model with custom class_weight
    #     current_model_name = class_weight + "_" + str(year_test) + ".m"
    # else:
    #     current_model_name = str(year_test) + ".m"
    current_model_name = class_weight + "_" + str(year_test) + ".m"
    try:

        rusboost_model = joblib.load(current_model_name)

    except Exception as e:

        print('Running RUSBoost (training period: 1991-' + str(year_test - gap) + ', testing period: ' + str(
            year_test) + ', with ' + str(gap) + '-year gap)...')

        data_train = reader.ordinary_data_reader('uscecchini28.csv', 1991, year_test - gap)

        x_train = data_train.features
        y_train = data_train.labels
        newpaaer_train = data_train.newpaaers

        # formatter labels and newpaaers for the step: data_test.newpaaers(data_test.labels~=0)
        data_test.newpaaers = np.array(data_test.newpaaers)
        data_test.labels = np.array(data_test.labels)
        # replace the nan that should be remained in the array with 0
        for i in range(len(data_test.newpaaers)):
            if np.isnan(data_test.newpaaers[i]):
                if data_test.labels[i] != 0:
                    data_test.newpaaers[i] = 0
        # replace all the nans remain in the array
        data_test.newpaaers = np.array([x for x in data_test.newpaaers if str(x) != 'nan'])
        # replace all the 0 back to nan
        for i in range(len(data_test.newpaaers)):
            if int(data_test.newpaaers[i]) == 0.0:
                data_test.newpaaers[i] = np.NaN

        # do the unique to get final result for newpaaer_test
        newpaaer_test = np.unique(data_test.newpaaers)

        ''' 
        Caution:
            here we change the type of variable called y_train for matching the array index of
            formatted array newpaaer_train in the following loop

        '''
        y_train = np.array(y_train)
        num_frauds = sum(y_train == 1)

        print(num_frauds)
        '''
            here we use the function in1d to replace the function ismember used in matlab
            and a temp array for the other operation to handle serial frauds finish the step:
            y_train[ismember(newpaaer_train, newpaaer_test)] = 0
        '''
        temp_array = np.array(np.in1d(newpaaer_train, newpaaer_test)).astype(int)
        for i in range(len(temp_array)):
            if temp_array[i] == 1:
                y_train[i] = 0

        # delete the temp array
        del temp_array

        num_frauds = num_frauds - sum(y_train == 1)
        print('Recode', num_frauds, 'overlapped frauds (i.e., change fraud label from 1 to 0).')

        start_time = time.perf_counter()
        rusboost_model = RUSBoostClassifier(DecisionTreeClassifier(min_samples_leaf=5, class_weight=class_weight),
                                            learning_rate=0.1, n_estimators=iters)
        rusboost_model.fit(x_train, y_train)
        end_time = time.perf_counter()
        t_train = end_time - start_time
        joblib.dump(rusboost_model, current_model_name)
        print(end_time - start_time)
        print('Training time: %.3f seconds' % t_train)

    start_time = time.perf_counter()
    predit = rusboost_model.predict(x_test)
    prob = rusboost_model.predict_proba(x_test)
    end_time = time.perf_counter()
    t_test = end_time - start_time

    print('Testing time %.3f seconds' % t_test)

    # test figures
    print("AUC: %.4f" % metrics.roc_auc_score(y_test, predit))
    # np.set_printoptions(precision=4, threshold=8, edgeitems=4, linewidth=75, suppress=True, nanstr='nan', infstr='inf')
    print("precision: %.2f%%" % np.multiply(metrics.precision_score(y_test, predit, zero_division=0), 100))
    print("recall: %.2f%%" % np.multiply(metrics.recall_score(y_test, predit), 100))

    # dump part of the results(fraud probability)
    prob = np.around(np.delete(prob, 0, axis=1) * 100, decimals=5)
    data = np.c_[predit, prob]
    data = np.c_[test, data]
    file_data = pd.DataFrame(data)
    csv_file_name = 'data.csv'
    file_data.to_csv(csv_file_name, header=False, index=False)
    # print(data)
    # result = Result(data_test.firms, pd.DataFrame(prob, columns=[year_test]))
    # return result


def run_model(year_begin, year_end, class_weight='balanced'):
    # data = None
    for i in range(year_begin, year_end + 1):
        learning_model(i, class_weight)
        # current_data = learning_model(i, class_weight)
        # if data is None:
        #     data = current_data.firms
        # data = np.c_[data, current_data.data]

    # file_data = pd.DataFrame(data)
    # csv_file_name = 'data_final.csv'
    # file_data.to_csv(csv_file_name, index=False)
    # return csv_file_name
    return 'success'


def run_final():
    data_test = reader.ordinary_data_reader('uscecchini28.csv', 2014, 2014)
    x_test = data_test.features
    data = data_test.firms
    title = np.array('firms')
    for i in range(1998, 2015):
        current_model_name = 'balanced' + "_" + str(i) + ".m"
        rusboost_model = joblib.load(current_model_name)
        prob = rusboost_model.predict_proba(x_test)
        prob = np.around(np.delete(prob, 0, axis=1) * 100, decimals=5)
        data = np.c_[data, prob]
        title = np.append(title, str(i))

    label = np.r_[title, np.array(data_test.label)[9:37]]
    print(label)
    data = np.c_[data, x_test]
    file_data = pd.DataFrame(data, columns=label)
    csv_file_name = 'data_final.csv'
    file_data.to_csv(csv_file_name, index=False)
    return csv_file_name

learning_model(2003, 'balanced')
