from distutils.command import clean

import matplotlib as matplotlib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import svm
# from sklearn.neural_nertwork import MLPClassifier
# from sklearn.metriocs import confusion_matrix, flassification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
# %matplotlib inline
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

    return info1, info2, alphabetData, greekAlphabetData


def plotAlphabet(lastColumn):
    lastColumn.value_counts().sort_index(ascending=True).plot(kind='bar',
                                                              rot=0)  # , colors = ['#FF0000', '#FF7F00','#FFFF00', '#00FF00','#0000FF', '#4B0082']

    # sns.countplot(alphabetLastColumn.value_counts())
    plt.show()


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

    return train_test_split(X, Y, test_size=0.2, random_state=42)


def calculateConfusionMatrix(yTest, prediction):
    return confusion_matrix(yTest, prediction)


def calculateClassificationReport(yTest, prediction):
    return classification_report(yTest, prediction)


# missing 3a
def exportToCSV(fileName, conFusionMatrix, classificationReport):
    # will need to discuss how to format files
    dataFrame = pd.DataFrame(conFusionMatrix)
    dataFrame.to_csv(r'output/'+fileName, sep=";")


def baselineDecisionTree(xTrain, xTest, yTrain, yTest, model):
    # A baseline Decision Tree using entropy as a decision criterion and using default values for the rest of the parameters

    base_DT = tree.DecisionTreeClassifier(criterion='entropy')          # construct DT classifier with entropy criterion
    base_DT = base_DT.fit(xTrain, yTrain)                               # train the algorithm with training datasets
    base_DT_prediction = base_DT.predict(xTest)                         # make predictions on our test dataset

    # export DT to pdf using Graphviz (optional)
    dot_data = tree.export_graphviz(base_DT, out_file=None)
    graph = graphviz.Source(dot_data)
    graph.render('Base-DT-DS2')

    cfm = calculateConfusionMatrix(yTest, base_DT_prediction)
    cr = calculateClassificationReport(yTest, base_DT_prediction)
    # print(cfm)
    # print(cr)
    exportToCSV(model, cfm, cr)


class Main:
    info1, info2, alphabetData, greekAlphabetData = readCSV()

    lastColumn = getReplacedLastColumn(info2, greekAlphabetData)

    # plotAlphabet(lastColumn)

    xTrain, xTest, yTrain, yTest = naimursMassage(greekAlphabetData, lastColumn)

    print(xTest)

    baselineDecisionTree(xTrain, xTest, yTrain, yTest, 'Base-DT-DS2')