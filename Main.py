from distutils.command import clean

import matplotlib as matplotlib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from sklearn import tree
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
import os


#CSV functions
def readCSV():
    info1 = pd.read_csv('dataset/info_1.csv')
    info2 = pd.read_csv('dataset/info_2.csv')

    alphabetData = pd.read_csv('dataset/train_1.csv', header=None)
    greekAlphabetData = pd.read_csv('dataset/train_2.csv', header=None)

    alphabetDataTestLabel = pd.read_csv('dataset/test_with_label_1.csv', header=None)
    greekAlphabetDataTestLabel = pd.read_csv('dataset/test_with_label_2.csv', header=None)

    return info1, info2, alphabetData, greekAlphabetData, alphabetDataTestLabel, greekAlphabetDataTestLabel, getReverseDic(info1), getReverseDic(info2)


def exportToCSV(fileName, instance_predicted_class, conFusionMatrix, classificationReport):
    cf = pd.DataFrame(conFusionMatrix)
    cr = pd.DataFrame(classificationReport).transpose()

    print(instance_predicted_class + "\n")
    print(cf.to_string() + "\n")
    print(cr.to_string() + "\n")

    with open(r'output/'+fileName, 'w') as f:
        f.write(instance_predicted_class)
    cf.to_csv(r'output/'+fileName, sep=',', mode='a')
    cr.to_csv(r'output/'+fileName, sep=',', mode='a')


#helper functions
def plotAlphabet(lastColumn):
    lastColumn.value_counts().sort_index(ascending=True).plot(kind='bar',
                                                              rot=0)  # , colors = ['#FF0000', '#FF7F00','#FFFF00', '#00FF00','#0000FF', '#4B0082']
    plt.show()

def getReverseDic(info):
    info_index = info['index'].tolist()
    info_symbol = info['symbol'].tolist()
    return dict(zip(info_symbol, info_index))

def getReplacedLastColumn(info, AlphabetData, lastColIndex=1024):
    info_index = info['index'].tolist()
    info_symbol = info['symbol'].tolist()
    infoDict = dict(zip(info_index, info_symbol))
    alphabetLastColumn = AlphabetData[AlphabetData.columns[lastColIndex]]
    alphabetLastColumn = alphabetLastColumn.replace(infoDict)

    return alphabetLastColumn

def naimursMassage(alphabetData, alphabetLastColumn, lastColIndex=1024):
    X = alphabetData.drop(alphabetData.columns[lastColIndex], axis=1)
    Y = alphabetLastColumn

    return X, Y


def calculateConfusionMatrix(yTest, prediction):
    return confusion_matrix(yTest, prediction)


def calculateClassificationReport(yTest, prediction):
    return classification_report(yTest, prediction, output_dict=True)


def instancePredictedClass(prediction, reverseDic):

    instance_predictedClass = ''

    i = 1

    for x in prediction:
        instance_predictedClass += str(i) + ',' + str(reverseDic.get(x)) + '\n'
        i += 1

    return instance_predictedClass

# number 2 a)
def GaussianNaiveBayes(xTrain, xTest, yTrain, yTest, reverseDic, model):
    gnb = GaussianNB()
    gnb = gnb.fit(xTrain, yTrain)
    gnb_prediction = gnb.predict(xTest)

    gnb_confusion_matrix = calculateConfusionMatrix(yTest, gnb_prediction)
    gnb_classification_report = calculateClassificationReport(yTest, gnb_prediction)
    gnb_predicted_class = instancePredictedClass(gnb_prediction, reverseDic)
    # print(gnb_confusion_matrix)

    exportToCSV(model, gnb_predicted_class, gnb_confusion_matrix, gnb_classification_report)


# number 2 b)
def baselineDecisionTree(xTrain, xTest, yTrain, yTest, reverseDic, model):
    # A baseline Decision Tree using entropy as a decision criterion and using default values for the rest of the parameters

    base_DT = tree.DecisionTreeClassifier(criterion='entropy')          # construct DT classifier with entropy criterion
    base_DT = base_DT.fit(xTrain, yTrain)                               # train the algorithm with training datasets
    base_DT_prediction = base_DT.predict(xTest)                         # make predictions on our test dataset
    base_DT_predicted_class = instancePredictedClass(base_DT_prediction, reverseDic)

    cfm = calculateConfusionMatrix(yTest, base_DT_prediction)
    cr = calculateClassificationReport(yTest, base_DT_prediction)

    # print(cfm)
    # print(cr)
    #
    # print(base_DT_predicted_class)
    exportToCSV(model, base_DT_predicted_class, cfm, cr)

# number 2 c)
def betterPerformingDecisionTree(xTrain, xTest, yTrain, yTest, reverseDic, model):
    # A better performing Decision Tree found by performing grid search to find the best combination of hyper-parameters.
    base_DT = tree.DecisionTreeClassifier()          # construct DT classifier with entropy criterion
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10],
        'min_samples_split': [2, 3, 4, 5, 6],
        'min_impurity_decrease': [0.0, 0.5, 1.0, 1.5, 2.0],
        'class_weight': [None, 'balanced']
    }
    search = GridSearchCV(base_DT, param_grid, n_jobs=-1, cv=5)
    search.fit(xTrain, yTrain)

    # print('Best Criterion:', search.best_estimator_.get_params()['criterion'])
    # print('Best max_depth:', search.best_estimator_.get_params()['max_depth'])
    # print('Best min_samples_split:', search.best_estimator_.get_params()['min_samples_split'])
    # print('Best min_impurity_decrease:', search.best_estimator_.get_params()['min_impurity_decrease'])
    # print('Best class_weight:', search.best_estimator_.get_params()['class_weight'])

    # create best_dt from all the best params found in the grid search
    best_DT = tree.DecisionTreeClassifier(criterion=str(search.best_estimator_.get_params()['criterion']),
                                          max_depth=search.best_estimator_.get_params()['max_depth'],
                                          min_samples_split=search.best_estimator_.get_params()['min_samples_split'],
                                          min_impurity_decrease=search.best_estimator_.get_params()['min_impurity_decrease'],
                                          class_weight=str(search.best_estimator_.get_params()['class_weight']))
    best_DT = best_DT.fit(xTrain, yTrain)                                                       # train the algorithm with training datasets
    best_DT_prediction = best_DT.predict(xTest)                                                 # make predictions
    best_DT_predicted_class = instancePredictedClass(best_DT_prediction, reverseDic)

    cfm = calculateConfusionMatrix(yTest, best_DT_prediction)
    cr = calculateClassificationReport(yTest, best_DT_prediction)

    # print(best_DT_predicted_class)
    # print(cfm)
    # print(cr)

    exportToCSV(model, best_DT_predicted_class, cfm, cr)

# number 2 d)
def classifyPerceptron(xTrain, xTest, yTrain, yTest, reverseDic, model):

    clf = Perceptron(tol=1e-3, random_state=0)
    clf = clf.fit(xTrain, yTrain)
    clf_prediction = clf.predict(xTest)

    clf_confusion_matrix = calculateConfusionMatrix(yTest, clf_prediction)
    clf_classification_report = calculateClassificationReport(yTest, clf_prediction)
    clf_predicted_class = instancePredictedClass(clf_prediction, reverseDic)
    # print(clf_confusion_matrix)

    exportToCSV(model, clf_predicted_class, clf_confusion_matrix, clf_classification_report)


# number 2 e)
def baseMLP(xTrain, xTest, yTrain, yTest, reverseDic, model):
    # a baseline Multi-Layered Perceptron with 1 hidden layer of 100 neurons, sigmoid/logistic as activation function,
    # stochastic gradient descent, and default values for the rest of the parameters.

    base_MLP = MLPClassifier(hidden_layer_sizes=(100,), activation='logistic',
                             solver='sgd')  # construct MLP Classifier with parameters
    base_MLP = base_MLP.fit(xTrain, yTrain)  # train the algorithm with training datasets
    base_MLP_prediction = base_MLP.predict(xTest)  # make predictions on our test dataset

    cfm = calculateConfusionMatrix(yTest, base_MLP_prediction)
    cr = calculateClassificationReport(yTest, base_MLP_prediction)
    clf_predicted_class = instancePredictedClass(base_MLP_prediction, reverseDic)
    # print(cfm)
    # print(cr)
    # print(clf_predicted_class)

    exportToCSV(model, clf_predicted_class, cfm, cr)


# number 2 f)
def bestMLP(xTrain, xTest, yTrain, yTest, reverseDic, model):
    # a better performing Multi-Layered Perceptron found by performing grid search to find the
    # best combination of hyper-parameters.
    mlp_gs = MLPClassifier(max_iter=100)
    parameter_space = {
        'hidden_layer_sizes': [(30, 50), (10, 10, 10)],
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'solver': ['sgd', 'adam']
    }
    clf = GridSearchCV(mlp_gs, parameter_space, n_jobs=-1, cv=5)
    clf.fit(xTrain, yTrain)  # X is train samples and y is the corresponding labels


    # base_MLP = MLPClassifier(max_iter=100)  # construct MLP Classifier with parameters
    #
    # param_grid = {
    #
    #     'activation': ['identity', 'logistic', 'tanh', 'relu'],
    #     'solver': ['sgd', 'adam']
    # }
    # search = GridSearchCV(base_MLP, param_grid, n_jobs=-1)
    # search.fit(xTrain, yTrain)

    # For this, you need to experiment with the following parameter values:
    #  • activation function: sigmoid, tanh, relu and identity
    #  • 2 network architectures of your choice: for eg 2 hidden layers with 30+50 nodes, 3 hidden layers with 10+10
    #  • solver: Adam and stochastic gradient descent

    #'hidden_layer_sizes': [(30, 50), (10, 10, 10)],
    #hidden_layer_sizes = str(search.best_estimator_.get_params()['hidden_layer_sizes']),
    # bestMLP = MLPClassifier(
    #                         activation=search.best_estimator_.get_params()['activation'],
    #                         solver=search.best_estimator_.get_params()['solver'])
    # bestMLP = bestMLP.fit(xTrain, yTrain)  # train the algorithm with training datasets
    best_MLP_prediction = clf.predict(xTest)  # make predictions on our test dataset

    cfm = calculateConfusionMatrix(yTest, best_MLP_prediction)
    cr = calculateClassificationReport(yTest, best_MLP_prediction)
    clf_predicted_class = instancePredictedClass(best_MLP_prediction, reverseDic)
    # print(cfm)
    # print(cr)
    # print(clf_predicted_class)

    exportToCSV(model, clf_predicted_class, cfm, cr)

# main runner class
class Main:
    info1, info2, alphabetData, greekAlphabetData, alphabetDataTestLabel, greekAlphabetDataTestLabel, reverseAlphaDic, reverseGreekDic = readCSV()

    lastColumn = getReplacedLastColumn(info1, alphabetData)
    lastColumnTest = getReplacedLastColumn(info1, alphabetDataTestLabel)
    # plotAlphabet(lastColumn)

    xTrain, yTrain = naimursMassage(alphabetData, lastColumn)
    xTest, yTest = naimursMassage(alphabetDataTestLabel, lastColumnTest)

    # GaussianNaiveBayes(xTrain, xTest, yTrain, yTest, reverseAlphaDic, 'GNB-DS1')
    # baselineDecisionTree(xTrain, xTest, yTrain, yTest, reverseAlphaDic, 'BASE-DT-DS1')
    betterPerformingDecisionTree(xTrain, xTest, yTrain, yTest, reverseAlphaDic, 'Best-DT-DS1')
    # classifyPerceptron(xTrain, xTest, yTrain, yTest, reverseAlphaDic, 'PER-DS1')
    # baseMLP(xTrain, xTest, yTrain, yTest, reverseAlphaDic, 'Base-MLP-DS1')
    # bestMLP(xTrain, xTest, yTrain, yTest, reverseAlphaDic, 'Best-MLP-DS2')


    lastColumn = getReplacedLastColumn(info2, greekAlphabetData)
    lastColumnTest = getReplacedLastColumn(info2, greekAlphabetDataTestLabel)
    # plotAlphabet(lastColumn)

    xTrain, yTrain = naimursMassage(greekAlphabetData, lastColumn)
    xTest, yTest = naimursMassage(greekAlphabetDataTestLabel, lastColumnTest)

    # GaussianNaiveBayes(xTrain, xTest, yTrain, yTest, reverseGreekDic, 'GNB-DS2')
    # baselineDecisionTree(xTrain, xTest, yTrain, yTest, reverseGreekDic, 'BASE-DT-DS2')
    # betterPerformingDecisionTree(xTrain, xTest, yTrain, yTest, reverseGreekDic, 'Best-DT-DS2')
    # classifyPerceptron(xTrain, xTest, yTrain, yTest, reverseGreekDic, 'PER-DS2')
    # baseMLP(xTrain, xTest, yTrain, yTest, reverseGreekDic, 'Base-MLP-DS2')
    # bestMLP(xTrain, xTest, yTrain, yTest, reverseGreekDic, 'Best-MLP-DS2')