import warnings
import pandas as pd
import re
import math

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report

warnings.filterwarnings('ignore')


# CSV functions
def readCSV():
    training_csv = pd.read_csv('dataset/covid_training.tsv', sep="\t", header=0)
    test_csv = pd.read_csv('dataset/covid_test_public.tsv', sep="\t")

    return training_csv, test_csv


# Returns data frame with only the columns we care about.
def TrimColumns(dataframe):
    dataframe = dataframe[["q1_label", "text"]]
    return dataframe


# Map q1_label to binary 0 or 1 for no and yes respectively
def convertAlphabeticClass(dataframe):
    mapper = {}

    for i, cat in enumerate(dataframe["q1_label"].unique()):
        mapper[cat] = i

    dataframe["q1_label"] = dataframe["q1_label"].map(mapper)
    return dataframe


# Cleans text, removes punctuation.
def CleanText(dataframe):
    dataframe['text'] = dataframe['text'].str.replace(
        '\W', ' ')  # Removes punctuation
    dataframe['text'] = dataframe['text'].str.lower()
    dataframe.head(3)

    return dataframe


# Gets all the unique words in the document
def GetVocabulary(dataframe, remove_words_appear_once=False):
    dataframe['text'] = dataframe['text'].str.split()

    vocabulary = []
    for tweet in dataframe['text']:
        for word in tweet:
            vocabulary.append(word)
    if remove_words_appear_once:
        vocabulary = [x for x in vocabulary if vocabulary.count(x) > 1]

    vocabulary = list(set(vocabulary))

    return vocabulary


# Get a dataframe with frequency of each word per message
def GetTokenizedDataframe(dataframe, vocabulary):
    word_counts_per_message = {unique_word: [0] * len(dataframe['text']) for unique_word in vocabulary}

    for index, tweet in enumerate(dataframe['text']):
        for word in tweet:
            if word in vocabulary:
                word_counts_per_message[word][index] += 1

    return pd.DataFrame(word_counts_per_message)


# Randomize the dataset
def RandomizeDataSet(dataframe):
    data_randomized = dataframe.sample(frac=1, random_state=1)

    # Calculate index for split
    training_test_index = round(len(data_randomized) * 0.8)

    # Split into training and test sets
    training_set = data_randomized[:training_test_index].reset_index(drop=True)
    test_set = data_randomized[training_test_index:].reset_index(drop=True)

    # print(training_set.shape)
    # print(test_set.shape)
    return training_set, test_set


# Calculating constants
def CalculateConstant(dataframe, vocabulary):
    # Isolating_no and yes messages first
    no_messages = dataframe[dataframe['q1_label'] == 0]
    yes_messages = dataframe[dataframe['q1_label'] == 1]

    # Calculate probability of no and yes classes
    p_no = len(no_messages) / len(dataframe)
    p_yes = len(yes_messages) / len(dataframe)

    # N_no
    n_words_per_no_message = no_messages['text'].apply(len)
    n_no = n_words_per_no_message.sum()

    # N_yes
    n_words_per_yes_message = yes_messages['text'].apply(len)
    n_yes = n_words_per_yes_message.sum()

    # N_Vocabulary
    n_vocabulary = len(vocabulary)

    # Laplace smoothing
    alpha = 0.01

    return no_messages, yes_messages, p_no, p_yes, n_no, n_yes, n_vocabulary, alpha


# Calculate parameters
def CalculateParameters(no_messages, yes_messages, alpha, n_no, n_yes, vocabulary, n_vocabulary):
    # Initiate parameters
    parameters_no = {unique_word: 0 for unique_word in vocabulary}
    parameters_yes = {unique_word: 0 for unique_word in vocabulary}

    # Calculate parameters
    for word in vocabulary:
        n_word_given_no = no_messages[word].sum()  # no_messages already defined
        n_word_given_no = (n_word_given_no + alpha) / (n_no + alpha * n_vocabulary)
        parameters_no[word] = n_word_given_no

        n_word_given_yes = yes_messages[word].sum()  # yes_messages already defined
        n_word_given_yes = (n_word_given_yes + alpha) / (n_yes + alpha * n_vocabulary)
        parameters_yes[word] = n_word_given_yes

    return parameters_no, parameters_yes


def predict(message, p_no, p_yes, parameters_no, parameters_yes, returnScore = False):
    message = re.sub('\W', ' ', message)
    message = message.lower().split()

    p_no_given_message = math.log(p_no)
    p_yes_given_message = math.log(p_yes)

    for word in message:
        if word in parameters_no:
            p_no_given_message += math.log(parameters_no[word])

        if word in parameters_yes:
            p_yes_given_message += math.log(parameters_yes[word])

    if p_yes_given_message > p_no_given_message:

        if returnScore:
            return p_yes_given_message
        else:
            return 'yes'
    elif p_no_given_message > p_yes_given_message:

        if returnScore:
            return p_no_given_message
        else:
            return 'no'
    else:
        return 'unknown'


# Gets evaluation metrics
def Evaluate(labels, predicted_labels):
    accuracy = accuracy_score(labels, predicted_labels)
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(labels, predicted_labels, pos_label="yes")
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(labels, predicted_labels, pos_label="yes")
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(labels, predicted_labels, pos_label="yes")
    print('F1 score: %f' % f1)


def writeEvaluationFile(filename, clr):

    with open(r'output/' + filename, 'w') as f:
        f.write(str(round(clr['accuracy'], 4)) + '\r')
        f.write(str(round(clr['Yes']['precision'], 4)) + '  ' + str(round(clr['No']['precision'], 4)) + '\r')
        f.write(str(round(clr['Yes']['recall'], 4)) + '  ' + str(round(clr['No']['recall'], 4)) + '\r')
        f.write(str(round(clr['Yes']['f1-score'], 4)) + '  ' + str(round(clr['No']['f1-score'], 4)) + '\r')
        f.close()

def writeTraceFile(filename, test_csv):

    with open(r'output/' + filename, 'w') as f:

        for x in range(len(test_csv)):
            # f.write(str(test_csv['tweet_id'][x]))
            f.write(str(test_csv['tweet_id'][x]) + '  ' + str(test_csv['predicted'][x]) + '  ' + "{:e}".format(float(test_csv['score'][x])) + '  ' + str(test_csv['q1_label'][x]) + '  ' + str(test_csv['correct'][x]) + '\r')

        f.close()

# main runner class
class Main:
    # pd.set_option('display.max_colwidth', -1)
    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.width', None)
    # pd.set_option('display.max_colwidth', None)

    training_csv, test_csv, = readCSV()
    test_csv.columns = ['tweet_id', 'text', 'q1_label', 'q2_label', 'q3_label', 'q4label', 'q5_label', 'q6_label',
                        'q7_label']

    training_dataframe = convertAlphabeticClass(TrimColumns(training_csv))
    training_dataframe = CleanText(training_dataframe)

    # training_set, test_set = RandomizeDataSet(training_dataframe)

    vocabulary = GetVocabulary(training_dataframe, remove_words_appear_once=False)
    word_counts_per_message = GetTokenizedDataframe(training_dataframe, vocabulary)
    training_dataframe = pd.concat([training_dataframe, word_counts_per_message], axis=1)
    no_messages, yes_messages, p_no, p_yes, n_no, n_yes, n_vocabulary, alpha = CalculateConstant(training_dataframe,
                                                                                                 vocabulary)

    parameters_no, parameters_yes = CalculateParameters(no_messages, yes_messages, alpha, n_no, n_yes, vocabulary,
                                                        n_vocabulary)

    test_csv['predicted'] = test_csv[test_csv.columns[1]].apply(predict,
                                                                args=(p_no, p_yes, parameters_no, parameters_yes))

    test_csv['score'] = test_csv[test_csv.columns[1]].apply(predict,
                                                                args=(p_no, p_yes, parameters_no, parameters_yes, True))

    correctList = []

    for x in range(len(test_csv)):
        if test_csv['predicted'][x] == test_csv['q1_label'][x]:
            correctList.append('correct')
        else:
            correctList.append('wrong')

    test_csv['correct'] = correctList

    Evaluate(test_csv['q1_label'], test_csv['predicted'])

    print(test_csv[['tweet_id', 'predicted', 'score', 'q1_label', 'correct']].head(100))

    clr = classification_report(test_csv['q1_label'], test_csv['predicted'], target_names=['No', 'Yes'], output_dict= True)

    writeEvaluationFile('eval_NB-BOW-OV.txt', clr)
    writeTraceFile('trace_NB-BOW-OV.txt', test_csv)


