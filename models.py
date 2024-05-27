from sklearn.linear_model import Lasso, Ridge, LinearRegression, LogisticRegression
from sklearn.svm import SVC
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn import metrics
import time
from pickle import dump
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV


def Lasso_Regression(X_train, X_test, y_train, y_test):
    print('Lasso Regression:')
    print('-----------------')
    f = open('lasso.pkl', 'wb')
    lasso = Lasso(alpha=0.001)
    # Record the start time
    start_time = time.time()

    lasso.fit(X_train, y_train)
    # Record the end time
    end_time = time.time()
    # Calculate training time
    training_time = end_time - start_time
    y_pred = lasso.predict(X_test)
    dump(lasso, f)
    print('MSE : ' + str(metrics.mean_squared_error(y_test, y_pred)))
    print('R2 Score : ' + str(metrics.r2_score(y_test, y_pred)))
    print(f'Training Time: {training_time} seconds')
    print('----------------------------------------------------------------------------------')


def Ridge_Regression(X_train, X_test, y_train, y_test):
    print('Ridge Regression:')
    print('-----------------')
    f = open('ridge.pkl', 'wb')
    ridge = Ridge(alpha=0.7)
    # Record the start time
    start_time = time.time()
    ridge.fit(X_train, y_train)
    # Record the end time
    end_time = time.time()
    # Calculate training time
    training_time = end_time - start_time
    y_pred = ridge.predict(X_test)
    dump(ridge, f)
    print('MSE : ' + str(metrics.mean_squared_error(y_test, y_pred)))
    print('R2 Score : ' + str(metrics.r2_score(y_test, y_pred)))
    print(f'Training Time: {training_time} seconds')
    print('----------------------------------------------------------------------------------')


def Polynomial_Regression(X_train, X_test, y_train, y_test):
    print('Polynomial Regression:')
    print('-----------------')
    f = open('degree.pkl', 'wb')
    f2 = open('poly.pkl', 'wb')
    poly_features = PolynomialFeatures(degree=2)
    # Record the start time
    start_time = time.time()
    X_train_poly = poly_features.fit_transform(X_train)
    dump(poly_features, f)
    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train)
    # Record the end time
    end_time = time.time()
    # Calculate training time
    training_time = end_time - start_time
    prediction = poly_model.predict(poly_features.fit_transform(X_test))
    dump(poly_model, f2)
    print('MSE : ' + str(metrics.mean_squared_error(y_test, prediction)))
    print('R2 Score : ' + str(metrics.r2_score(y_test, prediction)))
    print(f'Training Time: {training_time} seconds')
    print('----------------------------------------------------------------------------------')


def Linear_Regression(X_train, X_test, y_train, y_test):
    print('Linear Regression:')
    print('-----------------')
    f = open('linear.pkl', 'wb')
    start_time = time.time()
    model = LinearRegression()
    model.fit(X_train, y_train)
    # Record the end time
    end_time = time.time()
    # Calculate training time
    training_time = end_time - start_time
    ypred = model.predict(X_test)
    dump(model, f)
    print('MSE : ' + str(metrics.mean_squared_error(y_test, ypred)))
    print('R2 Score : ' + str(metrics.r2_score(y_test, ypred)))
    print(f'Training Time: {training_time} seconds')
    print('----------------------------------------------------------------------------------')


def SVM(X_train, X_test, y_train, y_test):
    print('Support Vector Machine:')
    print('-----------------------')
    f = open('svm.pkl', 'wb')
    svm = SVC(C=100, kernel='linear', gamma=0.01)
    # Record the start time of training
    start_time_train = time.time()
    svm.fit(X_train, y_train)
    # Record the end time of training
    end_time_train = time.time()
    # Record the start time of testing
    start_time_test = time.time()
    y_pred = svm.predict(X_test)
    # Record the end time of testing
    end_time_test = time.time()
    training_time = end_time_train - start_time_train
    testing_time = end_time_test - start_time_test
    dump(svm, f)
    # Confusion Matrix
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
    cm_display.plot()
    plt.title('Support Vector Machine')
    plt.show()
    print("Accuracy : ", metrics.accuracy_score(y_test, y_pred))
    print(f'Training Time: {training_time} seconds')
    print(f'Testing Time: {testing_time} seconds')
    print('----------------------------------------------------------------------------------')


def Logistic_Regression(X_train, X_test, y_train, y_test):
    print('Logistic Regression:')
    print('--------------------')
    f = open('logistic.pkl', 'wb')
    # train a logistic regression classifier using one-vs-one strategy
    lr = LogisticRegression(C=100, penalty='l2', solver='newton-cg')
    # Record the start time of training
    start_time_train = time.time()
    lr.fit(X_train, y_train)
    # Record the end time of training
    end_time_train = time.time()
    # Record the start time of testing
    start_time_test = time.time()
    # make predictions on test data
    y_pred = lr.predict(X_test)
    # Record the end time of testing
    end_time_test = time.time()
    training_time = end_time_train - start_time_train
    testing_time = end_time_test - start_time_test
    dump(lr, f)
    # Confusion Matrix
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
    cm_display.plot()
    plt.title('Logistic Regression')
    plt.show()
    print("Accuracy :", metrics.accuracy_score(y_test, y_pred))
    print(f'Training Time: {training_time} seconds')
    print(f'Testing Time: {testing_time} seconds')
    print('----------------------------------------------------------------------------------')


def KNN(X_train, X_test, y_train, y_test):
    print('K Nearest Neighbours:')
    print('---------------------')
    f = open('knn.pkl', 'wb')
    knn = KNeighborsClassifier(n_neighbors=5, weights='uniform')
    # Record the start time of training
    start_time_train = time.time()
    knn.fit(X_train, y_train)
    # Record the end time of training
    end_time_train = time.time()
    # Record the start time of testing
    start_time_test = time.time()
    y_pred = knn.predict(X_test)
    # Record the end time of testing
    end_time_test = time.time()
    training_time = end_time_train - start_time_train
    testing_time = end_time_test - start_time_test
    dump(knn, f)
    # Confusion Matrix
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
    cm_display.plot()
    plt.title('K Nearest Neighbours')
    plt.show()
    print("Accuracy : ", metrics.accuracy_score(y_test, y_pred))
    print(f'Training Time: {training_time} seconds')
    print(f'Testing Time: {testing_time} seconds')
    print('----------------------------------------------------------------------------------')


def random_forest(X_train, X_test, y_train, y_test):
    print('Random Forest:')
    print('---------------------')
    f = open('rf.pkl', 'wb')
    rf = RandomForestClassifier(random_state=42)
    # Record the start time of training
    start_time_train = time.time()
    rf.fit(X_train, y_train)
    # Record the end time of training
    end_time_train = time.time()
    # Record the start time of testing
    start_time_test = time.time()
    y_pred = rf.predict(X_test)
    # Record the end time of testing
    end_time_test = time.time()
    training_time = end_time_train - start_time_train
    testing_time = end_time_test - start_time_test
    dump(rf, f)
    # Confusion Matrix
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
    cm_display.plot()
    plt.title('Random Forest')
    plt.show()
    print("Accuracy : ", metrics.accuracy_score(y_test, y_pred))
    print(f'Training Time: {training_time} seconds')
    print(f'Testing Time: {testing_time} seconds')
    print('----------------------------------------------------------------------------------')
