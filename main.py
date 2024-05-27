from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
import tkinter as tk
from tkinter import ttk
from PIL import ImageTk, Image
from tkinter import filedialog
from preprocessing import *
from models import *


def train():
    # Milestone 1:
    print('Regression:')
    print('-----------')
    Car_Data = pd.read_csv("CarPrice_Milestone1.csv")
    Car_Data = fill_fueltype(Car_Data)
    Car_Data.dropna(inplace=True)
    X = Car_Data.iloc[:, 0:-1]  # Features
    Y = Car_Data.iloc[:, -1]  # Label

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=10)

    cols = ('symboling', 'CarName', 'fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel', 'enginelocation',
            'enginetype', 'cylindernumber', 'fuelsystem')
    X_train = OneHotEncoding_for_training(X_train, cols, "reg")
    X_test = OneHotEncoding_for_testing(X_test, "reg")

    training_data = pd.concat([X_train, y_train], axis=1, join="inner")

    training_data = Feature_selection_training(training_data, 0.6, "reg")
    X_test = Feature_selection_testing(X_test, "reg")

    testing_data = pd.concat([X_test, y_test], axis=1, join="inner")

    if varCheck.get() == 1:
        training_data = Remove_outliers(training_data, cols)

    training_data = Feature_scaling_for_training(training_data, "reg")
    testing_data = Feature_scaling_for_testing(testing_data, "reg")

    X_train = training_data.iloc[:, 0:-1]  # Features
    y_train = training_data.iloc[:, -1]  # Label

    X_test = testing_data.iloc[:, 0:-1]  # Features
    y_test = testing_data.iloc[:, -1]  # Label

    Lasso_Regression(X_train, X_test, y_train, y_test)

    Ridge_Regression(X_train, X_test, y_train, y_test)

    Polynomial_Regression(X_train, X_test, y_train, y_test)

    Linear_Regression(X_train, X_test, y_train, y_test)

    print('===========================================================================================================')

    # Milestone 2:
    print('Classification:')
    print('---------------')
    Classification_Data = pd.read_csv("CarPrice_Milestone2.csv")
    Classification_Data = fill_fueltype(Classification_Data)
    #Classification_Data.dropna(inplace=True)

    cols = ('symboling', 'CarName', 'fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel', 'enginelocation',
            'enginetype', 'cylindernumber', 'fuelsystem', 'category')
    Classification_Data = filling_missing_values(Classification_Data, cols)

    X = Classification_Data.iloc[:, 0:-1]  # Features
    Y = Classification_Data.iloc[:, -1]  # Label

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=10)

    training_data = pd.concat([X_train, y_train], axis=1, join="inner")
    testing_data = pd.concat([X_test, y_test], axis=1, join="inner")

    training_data = OneHotEncoding_for_training(training_data, cols, "class")
    testing_data = OneHotEncoding_for_testing(testing_data, "class")

    training_data = Feature_selection_training(training_data, 0.2, "class")

    X_test = testing_data.iloc[:, 0:-1]  # Features
    y_test = testing_data.iloc[:, -1]  # Label

    X_test = Feature_selection_testing(X_test, "class")

    if varCheck.get() == 1:
        training_data = Remove_outliers(training_data, cols)

    training_data = Feature_scaling_for_training(training_data, "class")

    testing_data = pd.concat([X_test, y_test], axis=1, join="inner")

    testing_data = Feature_scaling_for_testing(testing_data, "class")

    X_train = training_data.iloc[:, 0:-1]  # Features
    y_train = training_data.iloc[:, -1]  # Label

    X_test = testing_data.iloc[:, 0:-1]  # Features
    y_test = testing_data.iloc[:, -1]  # Label

    SVM(X_train, X_test, y_train, y_test)

    KNN(X_train, X_test, y_train, y_test)

    Logistic_Regression(X_train, X_test, y_train, y_test)

    random_forest(X_train, X_test, y_train, y_test)


def test():
    file_path = path_entry.get()

    data = pd.read_csv(file_path)
    data = fill_fueltype(data)

    if var.get() == "Regression":
        cols = ('symboling', 'CarName', 'fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel',
                'enginelocation', 'enginetype', 'cylindernumber', 'fuelsystem')
        regression_data = filling_missing_values(data, cols)
        x_test = regression_data.iloc[:, 0:-1]  # Features
        y_test = regression_data.iloc[:, -1]  # Label
        x_test, y_test = preprocesing_test_script(x_test, y_test, "reg")
        lasso1 = open('lasso.pkl', 'rb')
        ridge1 = open('ridge.pkl', 'rb')
        degree1 = open('degree.pkl', 'rb')
        poly1 = open('poly.pkl', 'rb')
        linear1 = open('linear.pkl', 'rb')
        # Lasso
        lasso = load(lasso1)
        print('Lasso Regression:')
        print('-----------------')
        y_pred = lasso.predict(x_test)
        print('MSE : ' + str(metrics.mean_squared_error(y_test, y_pred)))
        print('R2 Score : ' + str(metrics.r2_score(y_test, y_pred)))
        print('----------------------------------------------------------------------------------')
        lasso1.close()
        # Ridge
        ridge = load(ridge1)
        print('Ridge Regression:')
        print('-----------------')
        y_pred = ridge.predict(x_test)
        print('MSE : ' + str(metrics.mean_squared_error(y_test, y_pred)))
        print('R2 Score : ' + str(metrics.r2_score(y_test, y_pred)))
        print('----------------------------------------------------------------------------------')
        ridge1.close()
        # Polynomial
        degree = load(degree1)
        x_test2 = degree.transform(x_test)
        degree1.close()
        poly = load(poly1)
        print('Polynomial Regression:')
        print('-----------------')
        y_pred = poly.predict(x_test2)
        print('MSE : ' + str(metrics.mean_squared_error(y_test, y_pred)))
        print('R2 Score : ' + str(metrics.r2_score(y_test, y_pred)))
        print('----------------------------------------------------------------------------------')
        poly1.close()
        # Linear
        linear = load(linear1)
        print('Linear Regression:')
        print('-----------------')
        y_pred = linear.predict(x_test)
        print('MSE : ' + str(metrics.mean_squared_error(y_test, y_pred)))
        print('R2 Score : ' + str(metrics.r2_score(y_test, y_pred)))
        print('----------------------------------------------------------------------------------')
        linear1.close()
    else:
        cols = ('symboling', 'CarName', 'fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel',
                'enginelocation', 'enginetype', 'cylindernumber', 'fuelsystem', 'category')
        classification_data = filling_missing_values(data, cols)
        x_test = classification_data.iloc[:, 0:-1]  # Features
        y_test = classification_data.iloc[:, -1]  # Label
        x_test, y_test = preprocesing_test_script(x_test, y_test, "class")
        svm1 = open('svm.pkl', 'rb')
        lr1 = open('logistic.pkl', 'rb')
        knn1 = open('knn.pkl', 'rb')
        rf1 = open('rf.pkl', 'rb')
        # SVM
        svm = load(svm1)
        print('Support Vector Machine:')
        print('-----------------------')
        y_pred = svm.predict(x_test)
        print("Accuracy : ", metrics.accuracy_score(y_test, y_pred))
        print('----------------------------------------------------------------------------------')
        svm1.close()
        # Logistic Regression
        lr = load(lr1)
        print('Logistic Regression:')
        print('--------------')
        y_pred = lr.predict(x_test)
        print("Accuracy : ", metrics.accuracy_score(y_test, y_pred))
        print('----------------------------------------------------------------------------------')
        lr1.close()
        # KNN
        knn = load(knn1)
        print('K Nearest Neighbours:')
        print('---------------------')
        y_pred = knn.predict(x_test)
        print("Accuracy : ", metrics.accuracy_score(y_test, y_pred))
        print('----------------------------------------------------------------------------------')
        knn1.close()
        # Random Forest
        rf = load(rf1)
        print('K Nearest Neighbours:')
        print('---------------------')
        y_pred = rf.predict(x_test)
        print("Accuracy : ", metrics.accuracy_score(y_test, y_pred))
        print('----------------------------------------------------------------------------------')
        rf1.close()


def center_window(window, width, height):
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()

    x = (screen_width - width) // 2
    y = (screen_height - height) // 2

    window.geometry(f"{width}x{height}+{x}+{y}")


def gui():
    root = tk.Tk()
    root.title("Machine Learning Program")
    center_window(root, 800, 750)

    root.resizable(False, False)

    my_image = ImageTk.PhotoImage(file="Background.jpeg")

    canvas = tk.Canvas(root, bg='skyblue')
    canvas.create_text(400, 55, text="Welcome To Our Machine Learning Program", font=("times", 15, "bold"))
    canvas.create_text(400, 75, text="We Hope you Enjoy Your Time", font=("times", 11, "bold"))
    canvas.create_image(250, 120, image=my_image, anchor=tk.NW)
    canvas.create_text(380, 340, text="if  you want to train models\n       press on this button\n\t     ||\n\t     ||\n                    \  /", font=("times", 12, "bold"))
    canvas.create_text(350, 450, text="please choose the test script file:", font=("times", 12, "bold"))
    canvas.create_text(385, 670, text="test the new data script", font=("times", 13, "bold"))

    canvas.pack(fill="both", expand=True)

    choose_button = tk.Button(root, text="Choose", height=1, width=10, bg="#5DADE2", command=get_path)
    choose_button.place(x=480, y=440)

    train_button = tk.Button(root, text="Train", height=2, width=15, bg="#5DADE2", command=train)
    train_button.place(x=320, y=390)

    test_button = tk.Button(root, text="Test", height=2, width=15, bg="#5DADE2", command=test)
    test_button.place(x=330, y=690)

    global varCheck
    varCheck = tk.IntVar()
    check_box = tk.Checkbutton(root, text="Remove \nOutliers?", variable=varCheck, onvalue=1, offvalue=0,
                                    background='skyblue')
    check_box.place(x=290, y=350)

    global path_entry
    path_entry = tk.Entry(root, width=35)
    path_entry.place(x=240, y=470)

    s = ttk.Style()
    s.configure('Wild.TRadiobutton', background='skyblue', foreground='black')

    global var
    var = tk.StringVar()
    r1 = ttk.Radiobutton(root, text="Regression", variable=var, value="Regression", style="Wild.TRadiobutton")
    r1.place(x=250, y=530)

    r2 = ttk.Radiobutton(root, text="Classification", variable=var, value="Classification", style="Wild.TRadiobutton")
    r2.place(x=250, y=550)

    root.mainloop()


def get_path():
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.csv")])
    path_entry.delete(0)
    path_entry.insert(0, file_path)

    return file_path


gui()
#Test_Data = pd.read_csv(file_path)

#X_test = Classification_Data.iloc[:, 0:-1]  # Features
#Y_test = Classification_Data.iloc[:, -1]  # Label

#X_test, Y_test = preprocesing_test_script(X_test, Y_test, )
