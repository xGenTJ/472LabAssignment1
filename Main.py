import matplotlib as matplotlib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import svm
# from sklearn.neural_nertwork import MLPClassifier
# from sklearn.metriocs import confusion_matrix, flassification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
# %matplotlib inline
import os

class Main:

    info1 = pd.read_csv('dataset/info_1.csv')
    info2 = pd.read_csv('dataset/info_2.csv')

    info2_symbol = info2['symbol'].tolist()
    info1_symbol = info1['symbol'].tolist()
    print(info1_symbol)
    print(info2_symbol)