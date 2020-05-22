import pandas as pd
import numpy as np

'''
    here we define a class(Object) called Data to store the data read from the csv file
    and we trim all the useless code in switch-case model as the matlab function shows
    just keep the main code for reading csv file
'''


class Data:
    def __init__(self, label, years, firms, sics, insbnks, understatements, options, paaers, newpaaers, labels, features):
        self.label = label
        self.years = years
        self.firms = firms
        self.sics = sics
        self.insbnks = insbnks
        self.understatements = understatements
        self.options = options
        self.paaers = paaers
        self.newpaaers = newpaaers
        self.labels = labels
        self.features = features


def ordinary_data_reader(data_path, year_start, year_end):
    df = pd.read_csv(data_path)
    label = list(df.columns.values)
    df = df[df['fyear'] >= year_start]
    df = df[df['fyear'] <= year_end]
    years = df[label[0]]
    firms = df[label[1]]
    sics = df[label[2]]
    insbnks = df[label[3]]
    understatements = df[label[4]]
    options = df[label[5]]
    paaers = df[label[6]]
    newpaaers = df[label[7]]
    labels = df[label[8]]
    features = df[label[9]]
    for i in range(10, 37):
        a = label[i]
        features = np.c_[features, df[a]]

    num_observations = features.shape[0]
    num_features = features.shape[1]

    data = Data(label, years, firms, sics, insbnks, understatements, options, paaers, newpaaers, labels, features)
    print('Data Loaded: ' + data_path + ',', num_features, 'features,', num_observations,
          'observations (' + str(sum(labels == 1)) + ' pos, ' + str(sum(labels == 0)) + ' neg)')
    return data
