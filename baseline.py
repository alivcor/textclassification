from sklearn import datasets, neighbors, linear_model
from sklearn.feature_extraction.text import CountVectorizer
import os
import numpy as np
from numpy import array
import re
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import learning_curve
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from sklearn.datasets import load_files

def main():

    train_data_features, train_data_classes, test_data_features, actual_classes = getParams(1)

    model_name = "Logistic Regression - Unigram"
    trained_model, test_data_classes = train_logistic_regression(train_data_features, train_data_classes, test_data_features)
    evaluate_model(model_name, test_data_classes, actual_classes)
    cross_validate(model_name, trained_model, train_data_features, train_data_classes)

    model_name = "Naive Bayes - Unigram"
    trained_model, test_data_classes = train_naive_bayes(train_data_features, train_data_classes,test_data_features)
    evaluate_model(model_name, test_data_classes, actual_classes)
    cross_validate(model_name, trained_model, train_data_features, train_data_classes)

    model_name = "Random Forest - Unigram"
    trained_model, test_data_classes = train_random_forest(train_data_features, train_data_classes, test_data_features)
    evaluate_model(model_name, test_data_classes, actual_classes)
    cross_validate(model_name, trained_model, train_data_features, train_data_classes)

    model_name = "Support Vector Machines - Unigram"
    trained_model, test_data_classes = train_svm(train_data_features, train_data_classes, test_data_features)
    evaluate_model(model_name, test_data_classes, actual_classes)
    cross_validate(model_name, trained_model, train_data_features, train_data_classes)

    train_data_features, train_data_classes, test_data_features, actual_classes = getParams(2)

    model_name = "Logistic Regression - Bigram"
    trained_model, test_data_classes = train_logistic_regression(train_data_features, train_data_classes,test_data_features)
    evaluate_model(model_name, test_data_classes, actual_classes)
    cross_validate(model_name, trained_model, train_data_features, train_data_classes)

    model_name = "Naive Bayes - Bigram"
    trained_model, test_data_classes = train_naive_bayes(train_data_features, train_data_classes, test_data_features)
    evaluate_model(model_name, test_data_classes, actual_classes)
    cross_validate(model_name, trained_model, train_data_features, train_data_classes)
    #
    model_name = "Random Forest - Bigram"
    trained_model, test_data_classes = train_random_forest(train_data_features, train_data_classes, test_data_features)
    evaluate_model(model_name, test_data_classes, actual_classes)
    cross_validate(model_name, trained_model, train_data_features, train_data_classes)

    model_name = "Support Vector Machines - Bigram"
    trained_model, test_data_classes = train_svm(train_data_features, train_data_classes, test_data_features)
    evaluate_model(model_name, test_data_classes, actual_classes)
    cross_validate(model_name, trained_model, train_data_features, train_data_classes)


def read_data(ngram_val):
    data_sports = list()
    data_med = list()
    data_christ = list()
    data_misc = list()


    c_christ = 0
    c_med = 0
    c_misc = 0
    c_sports = 0

    dname = "dataset/Training/rec.sport.hockey/"
    for fname in os.listdir(dname):
        tfile = open(dname+fname, 'r')
        tdata = tfile.read()
        tdata = tdata.split("\n")
        tdata = tdata[8:len(tdata)]
        tdata = ' '.join(tdata)
        tdata = re.sub(r'[^\w]', ' ', tdata)
        tdata = re.sub(' +', ' ', tdata)
        data_sports.append(tdata)
        c_sports = c_sports + 1


    dname = "dataset/Training/sci.med/"
    for fname in os.listdir(dname):
        tfile = open(dname+fname, 'r')
        tdata = tfile.read()
        tdata = tdata.split("\n")
        tdata = tdata[8:len(tdata)]
        tdata = ' '.join(tdata)
        tdata = re.sub(r'[^\w]', ' ', tdata)
        tdata = re.sub(' +', ' ', tdata)
        data_med.append(tdata)
        c_med = c_med + 1


    dname = "dataset/Training/soc.religion.christian/"
    for fname in os.listdir(dname):
        tfile = open(dname+fname, 'r')
        tdata = tfile.read()
        tdata = tdata.split("\n")
        tdata = tdata[8:len(tdata)]
        tdata = ' '.join(tdata)
        tdata = re.sub(r'[^\w]', ' ', tdata)
        tdata = re.sub(' +', ' ', tdata)
        data_christ.append(tdata)
        c_christ = c_christ + 1


    dname = "dataset/Training/talk.religion.misc/"
    for fname in os.listdir(dname):
        tfile = open(dname+fname, 'r')
        tdata = tfile.read()
        tdata = tdata.split("\n")
        tdata = tdata[8:len(tdata)]
        tdata = ' '.join(tdata)
        tdata = re.sub(r'[^\w]', ' ', tdata)
        tdata = re.sub(' +', ' ', tdata)
        data_misc.append(tdata)
        c_misc = c_misc + 1

    vectorizer = CountVectorizer(analyzer="word", ngram_range =(ngram_val,ngram_val), tokenizer=None, preprocessor=None)

    all_train_data = data_sports + data_med + data_christ + data_misc
    train_data_features = (vectorizer.fit_transform(all_train_data)).toarray()


    # print train_data_features
    # print train_data_features.shape

    y_classes = [1]*c_sports + [2]*c_med + [3]*c_christ + [4]*c_misc

    train_data_classes = array(y_classes)

    # print train_data_classes
    # print train_data_classes.shape

    test_data_sports = list()
    test_data_med = list()
    test_data_christ = list()
    test_data_misc = list()

    test_c_christ = 0
    test_c_med = 0
    test_c_misc = 0
    test_c_sports = 0

    dname = "dataset/Test/rec.sport.hockey/"
    for fname in os.listdir(dname):
        tfile = open(dname + fname, 'r')
        tdata = tfile.read()
        tdata = tdata.split("\n")
        tdata = tdata[8:len(tdata)]
        tdata = ' '.join(tdata)
        tdata = re.sub(r'[^\w]', ' ', tdata)
        tdata = re.sub(' +', ' ', tdata)
        test_data_sports.append(tdata)
        test_c_sports = test_c_sports + 1

    dname = "dataset/Test/sci.med/"
    for fname in os.listdir(dname):
        tfile = open(dname + fname, 'r')
        tdata = tfile.read()
        tdata = tdata.split("\n")
        tdata = tdata[8:len(tdata)]
        tdata = ' '.join(tdata)
        tdata = re.sub(r'[^\w]', ' ', tdata)
        tdata = re.sub(' +', ' ', tdata)
        test_data_med.append(tdata)
        test_c_med = test_c_med + 1

    dname = "dataset/Test/soc.religion.christian/"
    for fname in os.listdir(dname):
        tfile = open(dname + fname, 'r')
        tdata = tfile.read()
        tdata = tdata.split("\n")
        tdata = tdata[8:len(tdata)]
        tdata = ' '.join(tdata)
        tdata = re.sub(r'[^\w]', ' ', tdata)
        tdata = re.sub(' +', ' ', tdata)
        test_data_christ.append(tdata)
        test_c_christ = test_c_christ + 1

    dname = "dataset/Test/talk.religion.misc/"
    for fname in os.listdir(dname):
        tfile = open(dname + fname, 'r')
        tdata = tfile.read()
        tdata = tdata.split("\n")
        tdata = tdata[8:len(tdata)]
        tdata = ' '.join(tdata)
        tdata = re.sub(r'[^\w]', ' ', tdata)
        tdata = re.sub(' +', ' ', tdata)
        test_data_misc.append(tdata)
        test_c_misc = test_c_misc + 1

    all_test_data = test_data_sports + test_data_med + test_data_christ + test_data_misc

    test_data_features = (vectorizer.transform(all_test_data)).toarray()

    # print test_data_features

    actual_classes = [1] * test_c_sports + [2]*test_c_med + [3]*test_c_christ + [4]*test_c_misc

    actual_classes = array(actual_classes)
    # print actual_classes

    return [train_data_features, train_data_classes, test_data_features, actual_classes]


def getParams(ngv):
    return read_data(ngv)


def train_logistic_regression(train_data_features, train_data_classes, test_data_features):
    np.random.seed(0)
    indices = np.arange(train_data_classes.shape[0])
    np.random.shuffle(indices)
    train_data_features, train_data_classes = train_data_features[indices], train_data_classes[indices]
    logreg = linear_model.LogisticRegression(C=1e5)
    logreg.fit(train_data_features, train_data_classes)
    test_data_classes = logreg.predict(test_data_features)
    return [logreg, test_data_classes]



def train_naive_bayes(train_data_features, train_data_classes, test_data_features):
    clf = GaussianNB()
    clf.fit(train_data_features, train_data_classes)
    test_data_classes = clf.predict(test_data_features)
    return [clf, test_data_classes]



def train_random_forest(train_data_features, train_data_classes, test_data_features):
    clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split = 2, random_state = 0)
    clf.fit(train_data_features, train_data_classes)
    test_data_classes = clf.predict(test_data_features)
    return [clf, test_data_classes]


def train_svm(train_data_features, train_data_classes, test_data_features):
    clf = svm.SVC()
    clf.fit(train_data_features, train_data_classes)
    test_data_classes = clf.predict(test_data_features)
    return [clf, test_data_classes]



def evaluate_model(model_name, test_data_classes, actual_classes):
    ascore = precision_score(actual_classes, test_data_classes, average=None)
    metrics = precision_recall_fscore_support(actual_classes, test_data_classes, average='macro')
    print model_name
    print "Precision Score : ",
    print ascore
    print "Precision : ", metrics[0],
    print "|| Recall : ", metrics[1],
    print "|| F-Score : ", metrics[2]
    print "------------------------------------"




def cross_validate(model_name, trained_model, train_data_features, train_data_classes):
    # Code for Validation Testing
    np.random.seed(0)
    indices = np.arange(train_data_classes.shape[0])
    np.random.shuffle(indices)

    train_data_features, train_data_classes = train_data_features[indices], train_data_classes[indices]
    maxk = len(train_data_classes)

    train_sizes, train_scores, valid_scores = learning_curve(trained_model, train_data_features, train_data_classes, train_sizes=[50, 100, 500, 1000, 1735], cv=5, scoring="f1_macro")

    print "Starting Validation Testing for ", model_name

    print "Train Sizes : ",
    print train_sizes
    print "Training Scores : ",
    print train_scores
    print "Validation Scores : ",
    print valid_scores

    print "~~~~~~~~~~~~~~~~~~~ Validation Complete ! Generating Plot ~~~~~~~~~~~~~~~~~~"
    test_scores = valid_scores
    title = "Learning Curves (" + model_name + ")"

    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("F1 Score")

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="F1 - Macro Average Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="F1 - Macro Average Cross-validation score")

    plt.legend(loc="best")
    plt.show()


if __name__ == '__main__':
    main()
