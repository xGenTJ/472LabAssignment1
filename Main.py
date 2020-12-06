import warnings
import pandas as pd
import re
import math

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
    for sms in dataframe['text']:
        for word in sms:
            vocabulary.append(word)
    if remove_words_appear_once:
        vocabulary = [x for x in vocabulary if vocabulary.count(x) > 1]

    vocabulary = list(set(vocabulary))

    return vocabulary


# Get a dataframe with frequency of each word per message
def GetTokenizedDataframe(dataframe, vocabulary):
    word_counts_per_message = {unique_word: [0] * len(dataframe['text']) for unique_word in vocabulary}

    for index, sms in enumerate(dataframe['text']):
        for word in sms:
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


def predict(message, p_no, p_yes, parameters_no, parameters_yes):
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
        return 'yes'
    elif p_no_given_message > p_yes_given_message:
        return 'no'
    else:
        return 'unknown'


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

    vocabulary = GetVocabulary(training_dataframe, True)
    word_counts_per_message = GetTokenizedDataframe(training_dataframe, vocabulary)
    training_dataframe = pd.concat([training_dataframe, word_counts_per_message], axis=1)
    no_messages, yes_messages, p_no, p_yes, n_no, n_yes, n_vocabulary, alpha = CalculateConstant(training_dataframe,
                                                                                                 vocabulary)

    parameters_no, parameters_yes = CalculateParameters(no_messages, yes_messages, alpha, n_no, n_yes, vocabulary,
                                                        n_vocabulary)

    test_csv['predicted'] = test_csv[test_csv.columns[1]].apply(predict,
                                                                args=(p_no, p_yes, parameters_no, parameters_yes))

    average_no = len(test_csv[test_csv['q1_label'] == "no"]) / len(test_csv[test_csv['predicted'] == "no"])
    print("Accuracy for no ", average_no)
    average_yes = len(test_csv[test_csv['q1_label'] == "yes"]) / len(test_csv[test_csv['predicted'] == "yes"])
    print("Accuracy for yes ", average_yes)

    print(test_csv[['text', 'q1_label', 'predicted']].head(100))
