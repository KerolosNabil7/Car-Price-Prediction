import pandas as pd
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, MinMaxScaler
from pickle import dump, load, dumps


def fill_fueltype(data):
    # drop nulls in compressionratio column
    data['compressionratio'] = data['compressionratio'].dropna()
    indecies = data['fueltype'].isna()
    index_of_null = []
    # get indecies of null values in fueltype column
    for j in range(len(indecies)):
        if indecies[j]:
            index_of_null.append(j)
    # prove that diesel for compressionratio > 20 and gas for compressionratio < 20
    list_diesel = []
    list_gas = []
    for k in range(len(data)):
        if k not in index_of_null:
            if data['compressionratio'][k] > 20:
                list_diesel.append(data['fueltype'][k])
            else:
                list_gas.append(data['fueltype'][k])
    if "diesel" in list_gas:
        print("The Rule Failed To Prove -_-")
    elif "gas" in list_diesel:
        print("The Rule Failed To Prove -_-")
    else:
        print("The Rule is Proved ;)")
    # fill fueltype column with diesel or gas
    for i in index_of_null:
        if float(data['compressionratio'][i]) > 20:
            data.at[i, 'fueltype'] = 'diesel'
        else:
            data.at[i, 'fueltype'] = 'gas'
    return data

def Remove_outliers(data, columns):
    cols = list(data.columns)
    cols.pop(-1)
    for i in columns:
        if i in cols:
            cols.remove(i)
    for col in cols:
        # IQR
        # Calculate the upper and lower limits
        data = data.reset_index(drop=True)
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        # Create arrays of Boolean values indicating the outlier rows
        upper_array = np.where(data[col] >= upper)[0]
        lower_array = np.where(data[col] <= lower)[0]

        # Removing the outliers
        data.drop(index=upper_array, inplace=True)
        data.drop(index=lower_array, inplace=True)
    print("Data Shape After Remove Outliers:", data.shape)
    return data

def OneHotEncoding_for_training(X_train, columns, model_type):
    # change the type of symboling column
    X_train['symboling'] = X_train['symboling'].astype(str)
    if model_type == 'reg':
        f = open('encoder1.pkl', 'wb')
        f2 = open('columns1.pkl', 'wb')
        dump(columns, f2)
    else:
        f = open('encoder2.pkl', 'wb')
        f2 = open('columns2.pkl', 'wb')
        dump(columns, f2)

    lbls = []
    for c in columns:
        # declare Label Encoder
        lbl = LabelEncoder()
        # fit and transform on training data columns
        lbl = lbl.fit(list(X_train[c].values))
        X_train[c] = lbl.transform(list(X_train[c].values))
        lbls.append(lbl)
    # save the encoders in pickle
    dump(lbls, f)
    f.close()
    return X_train

def OneHotEncoding_for_testing(X_test, model_type):
    # change the type of symboling column
    X_test['symboling'] = X_test['symboling'].astype(str)
    if model_type == 'reg':
        f = open('encoder1.pkl', 'rb')
        f2 = open('columns1.pkl', 'rb')
    else:
        f = open('encoder2.pkl', 'rb')
        f2 = open('columns2.pkl', 'rb')
    # load encoders
    columns = load(f2)
    lbls = load(f)
    i = 0
    for c in columns:
        lbl = lbls[i]

        X_test[c] = X_test[c].map(lambda s: '<others>' if s not in lbl.classes_ else s)
        lbl.classes_ = np.append(lbl.classes_, '<others>')
        # transform on testing data columns
        X_test[c] = lbl.transform(X_test[c])
        i = i + 1
    f.close()
    f2.close()
    return X_test

def Feature_selection_training(data, num, model_type):
    # perform correlation on all data
    corr = data.corr()
    # take the top features that made correlation score greater than num
    if model_type == 'reg':
        top_feature = corr.index[abs(corr['price']) > num]
        f = open('top_feature1.pkl', 'wb')
    else:
        top_feature = corr.index[abs(corr['category']) > num]
        f = open('top_feature2.pkl', 'wb')
    # plot the corelation score
    plt.subplots(figsize=(12, 8))
    top_corr = data[top_feature].corr()
    sns.heatmap(top_corr, annot=True)
    if model_type == "reg":
        plt.title('Regression')
    else:
        plt.title('Classification')
    plt.show()
    print("Number Of Columns Selected: " + str(len(top_feature)-1))
    data = data[top_feature]
    dump(top_feature, f)

    return data

def Feature_selection_testing(X_test, model_type):
    if model_type == 'reg':
        f = open('top_feature1.pkl', 'rb')
    else:
        f = open('top_feature2.pkl', 'rb')
    top_feature = load(f)
    top_feature = top_feature.delete(-1)
    X_test = X_test[top_feature]

    return X_test

def Feature_scaling_for_training(X_train, model_type):
    # declare MinMax for scaling
    scaler = MinMaxScaler()
    # scaling all features
    X_train[X_train.columns] = scaler.fit_transform(X_train[X_train.columns])
    if model_type == 'reg':
        f = open('scaler1.pkl', 'wb')
    else:
        f = open('scaler2.pkl', 'wb')
    dump(scaler, f)

    return X_train

def Feature_scaling_for_testing(X_test, model_type):
    # load MinMax for scaling from pickle
    if model_type == 'reg':
        f = open('scaler1.pkl', 'rb')
    else:
        f = open('scaler2.pkl', 'rb')
    scaler = load(f)
    # scaling all features
    X_test[X_test.columns] = scaler.transform(X_test[X_test.columns])

    return X_test

def filling_missing_values(data, cols):
    # Dealing with missing values in whole data
    for i in data:
        if i in cols:
            mode_value = data[i].mode()[0]
            data[i].fillna(mode_value, inplace=True)
            continue
        mean_value = data[i].mean()
        data[i].fillna(mean_value, inplace=True)
    return data

def preprocesing_test_script(x_test, y_test, model_type):
    if model_type == "reg":
        encoder1 = open('encoder1.pkl', 'rb')
        top_feature1 = open('top_feature1.pkl', 'rb')
        scaler1 = open('scaler1.pkl', 'rb')
        columns1 = open('columns1.pkl', 'rb')
        # saved encoding
        columns = load(columns1)
        lbls = load(encoder1)
        i = 0
        for c in columns:
            lbl = lbls[i]

            x_test[c] = x_test[c].map(lambda s: '<others>' if s not in lbl.classes_ else s)
            lbl.classes_ = np.append(lbl.classes_, '<others>')
            # transform on testing data columns
            x_test[c] = lbl.transform(x_test[c])
            i = i + 1
        columns1.close()
        encoder1.close()

    else:
        encoder1 = open('encoder2.pkl', 'rb')
        top_feature1 = open('top_feature2.pkl', 'rb')
        scaler1 = open('scaler2.pkl', 'rb')
        columns1 = open('columns2.pkl', 'rb')

        testing_data = pd.concat([x_test, y_test], axis=1, join="inner")
        # saved encoding
        columns = load(columns1)
        lbls = load(encoder1)
        i = 0
        for c in columns:
            lbl = lbls[i]

            testing_data[c] = testing_data[c].map(lambda s: '<others>' if s not in lbl.classes_ else s)
            lbl.classes_ = np.append(lbl.classes_, '<others>')
            # transform on testing data columns
            testing_data[c] = lbl.transform(testing_data[c])
            i = i + 1
        columns1.close()
        encoder1.close()
        x_test = testing_data.iloc[:, 0:-1]  # Features
        y_test = testing_data.iloc[:, -1]  # Label

    # saved feature selection
    top_feature = load(top_feature1)
    top_feature = top_feature.delete(-1)
    x_test = x_test[top_feature]
    top_feature1.close()

    testing_data = pd.concat([x_test, y_test], axis=1, join="inner")

    # saved scaling
    scaler = load(scaler1)
    # scaling all features
    testing_data[testing_data.columns] = scaler.transform(testing_data[testing_data.columns])

    x_test = testing_data.iloc[:, 0:-1]  # Features
    y_test = testing_data.iloc[:, -1]  # Label

    return x_test, y_test
