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
from sklearn.metrics import precision_recall_fscore_support as score
import os


# CSV functions
def readCSV():
    info1 = pd.read_csv('dataset/covid_training.tsv', sep="\t")
    info2 = pd.read_csv('dataset/covid_test_public.tsv', sep="\t")

    # alphabetData = pd.read_csv('dataset/train_1.csv', header=None)
    #
    # alphabetDataTestLabel = pd.read_csv('dataset/test_with_label_1.csv', header=None)
    #
    # alphaValidation = pd.read_csv('dataset/val_1.csv', header=None)
    # greekAlphaValidation = pd.read_csv('dataset/val_2.csv', header=None)

    return info1, info2


def exportToCSV(fileName, instance_predicted_class, conFusionMatrix, classificationReport):
    cf = pd.DataFrame(conFusionMatrix)
    cr = pd.DataFrame(classificationReport).transpose()

    # print(instance_predicted_class + "\n")
    # print(cf.to_string() + "\n")
    # print(cr.to_string() + "\n")

    with open(r'output/'+fileName, 'w') as f:
        f.write(instance_predicted_class)
    cf.to_csv(r'output/'+fileName, sep=',', mode='a')
    cr.to_csv(r'output/'+fileName, sep=',', mode='a')


def ExtractOriginalVocabulary(dataFrame):
    column = dataFrame["text"].to_numpy()
    original_vocab = Counter()
    for row in column:  # add the frequency of the words to the list
        li = list(row.split(" "))
        for word in li:
            if word != "":
                original_vocab[word.lower()] += 1
    filtered_vocab = {x: count for x, count in original_vocab.items() if count >= 2}

    # print(original_vocab.most_common())
    print(filtered_vocab.items())
    return original_vocab, filtered_vocab

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

def cleanUpData(alphabetData, alphabetLastColumn, lastColIndex=1024):
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
    precision, recall, fscore, support = score(yTest, gnb_prediction, average='macro')
    # print('Precision : {}'.format(precision))
    # print('Recall    : {}'.format(recall))
    # print('F-score   : {}'.format(fscore))
    # print('Support   : {}'.format(support))
    return gnb, fscore

# main runner class
class Main:
    info1, info2, alphabetData, greekAlphabetData, alphabetDataTestLabel, greekAlphabetDataTestLabel, reverseAlphaDic, reverseGreekDic, alphaValidation, greekAlphaValidation = readCSV()

