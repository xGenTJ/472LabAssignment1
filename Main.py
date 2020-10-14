from distutils.command import clean

import matplotlib as matplotlib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn import tree
import graphviz
from sklearn.metrics import classification_report, confusion_matrix
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz/bin/'

def readCSV():
    info1 = pd.read_csv('dataset/info_1.csv')
    info2 = pd.read_csv('dataset/info_2.csv')

    alphabetData = pd.read_csv('dataset/train_1.csv', header=None)
    greekAlphabetData = pd.read_csv('dataset/train_2.csv', header=None)

    alphabetDataTestLabel = pd.read_csv('dataset/test_with_label_1.csv', header=None)
    greekAlphabetDataTestLabel = pd.read_csv('dataset/test_with_label_2.csv', header=None)

    return info1, info2, alphabetData, greekAlphabetData, alphabetDataTestLabel, greekAlphabetDataTestLabel, getReverseDic(info1), getReverseDic(info2)

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
    return classification_report(yTest, prediction)

# missing 3a
def exportToCSV(fileName, instance_predicted_class, conFusionMatrix, classificationReport):


    # will need to discuss how to format files
    cf = pd.DataFrame(conFusionMatrix)

    with open(r'output/'+fileName, 'w') as f:
        f.write(instance_predicted_class + '\n\n')
        f.write(cf.to_string() + '\n\n')
        f.write(classificationReport + '\n\n')

    # cf.to_csv(r'output/'+fileName, sep=',')

def instancePredictedClass(prediction, reverseDic):

    instance_predictedClass = ''

    i = 1

    for x in prediction:
        instance_predictedClass += str(i) + ',' + str(reverseDic.get(x)) + '\n'
        i += 1

    return instance_predictedClass

def baselineDecisionTree(xTrain, xTest, yTrain, yTest, reverseDic, model):
    # A baseline Decision Tree using entropy as a decision criterion and using default values for the rest of the parameters

    base_DT = tree.DecisionTreeClassifier(criterion='entropy')          # construct DT classifier with entropy criterion
    base_DT = base_DT.fit(xTrain, yTrain)                               # train the algorithm with training datasets
    base_DT_prediction = base_DT.predict(xTest)                         # make predictions on our test dataset
    base_DT_predicted_class = instancePredictedClass(base_DT_prediction, reverseDic)
    # export DT to pdf using Graphviz (optional)
    # dot_data = tree.export_graphviz(base_DT, out_file=None)
    # graph = graphviz.Source(dot_data)
    # graph.render('Base-DT-DS2')

    cfm = calculateConfusionMatrix(yTest, base_DT_prediction)
    cr = calculateClassificationReport(yTest, base_DT_prediction)

    print(cfm)
    print(cr)

    print(base_DT_predicted_class)
    exportToCSV(model, base_DT_predicted_class, cfm, cr)

def classifyPerceptron(xTrain, xTest, yTrain, yTest, reverseDic, model):

    clf = Perceptron(tol=1e-3, random_state=0)
    clf = clf.fit(xTrain, yTrain)
    clf_prediction = clf.predict(xTest)

    clf_score = clf.score(xTrain, yTrain)
    clf_confusion_matrix = calculateConfusionMatrix(yTest, clf_prediction)
    clf_classification_report = calculateClassificationReport(yTest, clf_prediction)
    clf_predicted_class = instancePredictedClass(clf_prediction, reverseDic)
    print(clf_confusion_matrix)

    exportToCSV(model, instancePredictedClass(clf_prediction), clf_confusion_matrix, clf_classification_report)

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
    print(cfm)
    print(cr)
    print(clf_predicted_class)

    exportToCSV(model, clf_predicted_class, cfm, cr)

def bestMLP(xTrain, xTest, yTrain, yTest, reverseDic, model):
    # a better performing Multi-Layered Perceptron found by performing grid search to find the
    # best combination of hyper-parameters.

    # For this, you need to experiment with the following parameter values:
    #  • activation function: sigmoid, tanh, relu and identity
    #  • 2 network architectures of your choice: for eg 2 hidden layers with 30+50 nodes, 3 hidden layers with 10+10
    #  • solver: Adam and stochastic gradient descent

    base_MLP = MLPClassifier(hidden_layer_sizes=(30, 50), activation='logistic',
                             solver='sgd')  # construct MLP Classifier with parameters
    base_MLP = base_MLP.fit(xTrain, yTrain)  # train the algorithm with training datasets
    base_MLP_prediction = base_MLP.predict(xTest)  # make predictions on our test dataset

    cfm = calculateConfusionMatrix(yTest, base_MLP_prediction)
    cr = calculateClassificationReport(yTest, base_MLP_prediction)
    clf_predicted_class = instancePredictedClass(base_MLP_prediction, reverseDic)
    print(cfm)
    print(cr)
    print(clf_predicted_class)

    exportToCSV(model, clf_predicted_class, cfm, cr)

class Main:
    info1, info2, alphabetData, greekAlphabetData, alphabetDataTestLabel, greekAlphabetDataTestLabel, reverseAlphaDic, reverseGreekDic = readCSV()

    lastColumn = getReplacedLastColumn(info2, greekAlphabetData)
    lastColumnTest = getReplacedLastColumn(info2, greekAlphabetDataTestLabel)
    # plotAlphabet(lastColumn)

    xTrain, yTrain = naimursMassage(greekAlphabetData, lastColumn)
    xTest, yTest = naimursMassage(greekAlphabetDataTestLabel, lastColumnTest)


    # print(xTest)

    baselineDecisionTree(xTrain, xTest, yTrain, yTest, reverseGreekDic, 'Base-DT-DS2')

    # baseMLP(xTrain, xTest, yTrain, yTest, reverseGreekDic, 'Base-MLP-DS2')
    # bestMLP(xTrain, xTest, yTrain, yTest, reverseGreekDic, 'Best-MLP-DS2')