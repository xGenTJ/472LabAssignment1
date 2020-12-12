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


# Cleans text, removes punctuation and make everything lowercase.
def CleanText(dataframe):
    dataframe['text'] = dataframe['text'].str.replace(
        '\W', ' ')  # Removes punctuation
    dataframe['text'] = dataframe['text'].str.lower()

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


# Get a dataframe with frequency of each word per tweet
def GetTokenizedDataframe(dataframe, vocabulary):
    word_counts_per_tweet = {unique_word: [0] * len(dataframe['text']) for unique_word in vocabulary}

    for index, tweet in enumerate(dataframe['text']):
        for word in tweet:
            if word in vocabulary:
                word_counts_per_tweet[word][index] += 1

    return pd.DataFrame(word_counts_per_tweet)


# Calculating constants
# such as P(no), P(yes),
# number of words in all class no tweets,
# number of words in all class yes tweets,
# number of words in vocabulary.
def CalculateConstant(dataframe, vocabulary):
    # Separate tweets with yes and no classes
    no_tweets = dataframe[dataframe['q1_label'] == 0]
    yes_tweets = dataframe[dataframe['q1_label'] == 1]

    # Calculate probability of no and yes classes
    p_no = len(no_tweets) / len(dataframe)
    p_yes = len(yes_tweets) / len(dataframe)

    # number of words in all class no tweets
    n_words_per_no_tweet = no_tweets['text'].apply(len)
    n_no = n_words_per_no_tweet.sum()

    # number of words in all class yes tweets
    n_words_per_yes_tweet = yes_tweets['text'].apply(len)
    n_yes = n_words_per_yes_tweet.sum()

    # number of words in vocabulary
    n_vocabulary = len(vocabulary)

    # to smooth, use additive smoothing (add-δ) with δ = 0.01
    alpha = 0.01

    return no_tweets, yes_tweets, p_no, p_yes, n_no, n_yes, n_vocabulary, alpha


# Calculate conditional probability value associated
# with each word in the vocabulary
# for class no and yes
def CalculateConditionalProbabilities(no_tweets, yes_tweets, alpha, n_no, n_yes, vocabulary, n_vocabulary):
    # Set each conditional probability for class no and yes
    # to 0 for each unique word in vocabulary
    cond_probs_no = {unique_word: 0 for unique_word in vocabulary}
    cond_probs_yes = {unique_word: 0 for unique_word in vocabulary}

    # Calculate conditional probability for each word in vocabulary,
    # for both class no and yes with smoothing of alpha value
    for word in vocabulary:
        n_word_given_no = no_tweets[word].sum()
        n_word_given_no = (n_word_given_no + alpha) / (n_no + alpha * n_vocabulary)
        cond_probs_no[word] = n_word_given_no

        n_word_given_yes = yes_tweets[word].sum()
        n_word_given_yes = (n_word_given_yes + alpha) / (n_yes + alpha * n_vocabulary)
        cond_probs_yes[word] = n_word_given_yes

    return cond_probs_no, cond_probs_yes


# Classifies a tweet based on all parameters.
def predict(tweet, p_no, p_yes, cond_probs_no, cond_probs_yes, returnScore=False):
    # Remove punctuations and make everything lowercase
    tweet = re.sub('\W', ' ', tweet)
    tweet = tweet.lower().split()

    # Calculate the log value of class probabilities
    p_no_for_tweet = math.log(p_no)
    p_yes_for_tweet = math.log(p_yes)

    # Calculate the prediction score for each class
    for word in tweet:
        if word in cond_probs_no:
            p_no_for_tweet += math.log(cond_probs_no[word])

        if word in cond_probs_yes:
            p_yes_for_tweet += math.log(cond_probs_yes[word])

    # Compare the prediction score and return a class
    if p_yes_for_tweet > p_no_for_tweet:

        if returnScore:
            return p_yes_for_tweet
        else:
            return 'yes'
    elif p_no_for_tweet > p_yes_for_tweet:

        if returnScore:
            return p_no_for_tweet
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
            f.write(str(test_csv['tweet_id'][x]) + '  ' + str(test_csv['predicted'][x]) + '  ' + "{:e}".format(
                float(test_csv['score'][x])) + '  ' + str(test_csv['q1_label'][x]) + '  ' + str(
                test_csv['correct'][x]) + '\r')

        f.close()


# main runner class
class Main:

    training_csv, test_csv, = readCSV()
    test_csv.columns = ['tweet_id', 'text', 'q1_label', 'q2_label', 'q3_label', 'q4label', 'q5_label', 'q6_label',
                        'q7_label']

    training_dataframe = convertAlphabeticClass(TrimColumns(training_csv))
    training_dataframe = CleanText(training_dataframe)

    vocabulary = GetVocabulary(training_dataframe, remove_words_appear_once=False)
    word_counts_per_tweet = GetTokenizedDataframe(training_dataframe, vocabulary)
    # Create the dataframe with columns for each word and their frequencies
    training_dataframe = pd.concat([training_dataframe, word_counts_per_tweet], axis=1)
    no_tweets, yes_tweets, p_no, p_yes, n_no, n_yes, n_vocabulary, alpha = CalculateConstant(training_dataframe,
                                                                                             vocabulary)

    cond_probs_no, cond_probs_yes = CalculateConditionalProbabilities(no_tweets, yes_tweets, alpha, n_no, n_yes, vocabulary,
                                                                      n_vocabulary)

    test_csv['predicted'] = test_csv[test_csv.columns[1]].apply(predict,
                                                                args=(p_no, p_yes, cond_probs_no, cond_probs_yes))

    test_csv['score'] = test_csv[test_csv.columns[1]].apply(predict,
                                                            args=(p_no, p_yes, cond_probs_no, cond_probs_yes, True))

    correctList = []

    for x in range(len(test_csv)):
        if test_csv['predicted'][x] == test_csv['q1_label'][x]:
            correctList.append('correct')
        else:
            correctList.append('wrong')

    test_csv['correct'] = correctList

    Evaluate(test_csv['q1_label'], test_csv['predicted'])

    print(test_csv[['tweet_id', 'predicted', 'score', 'q1_label', 'correct']].head(100))

    clr = classification_report(test_csv['q1_label'], test_csv['predicted'], target_names=['No', 'Yes'],
                                output_dict=True)

    writeEvaluationFile('eval_NB-BOW-OV.txt', clr)
    writeTraceFile('trace_NB-BOW-OV.txt', test_csv)
