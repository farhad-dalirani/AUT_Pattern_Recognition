# Discrete Input
# Independent classifiers
# No parent
# Base Classifier: Naive Bayes
# Training Scheme: -
# Select Root: -
# No Chain


def create_classifiers(train):
    """
    This function creates classifiers for each class.
    This is an binary relevance method, so classifiers are
    independent.
    :param train: training data-set
    :return: a dictionary of trained classifiers
    """
    from sklearn.naive_bayes import BernoulliNB
    from read_inputs import create_training_set_for_binary_relevance_classifier

    # A dictionary of classifiers for each class
    # {class0: classifier0, ...,classN: classifierN}
    classifiers = {classifier: None for classifier in range(0, train['numberOfLabels'])}

    # for each class makes an gaussian naive bayes classifiers and train it
    for key in classifiers:
        classifiers[key] = BernoulliNB(alpha=1)

        # Create appropriate classifier
        trainingSet = create_training_set_for_binary_relevance_classifier(train=train,
                                                         classOfClassifier=key)

        # Train classifier
        classifiers[key].fit(trainingSet[0], trainingSet[1])

    return classifiers


def predict(classifiers, testSample):
    """
    Use classifiers to predict labels of input sample.
    :param classifiers: a dictionary of trained classifiers
    :param testSamples: A test sample which user wants
           to predict its labels
    :return: list of labels
    """
    # a dictionary for holding prediction for classes
    # it supposed 0 and -1 [maybe probability * should be checked]
    predictions = {Class: None for Class in classifiers}

    for _class in classifiers:

        label = classifiers[_class].predict([testSample])
        predictions[_class] = label[0]

    return predictions


def BRM_NB_DISCRETE(path):
    """
    This function uses other functions to read
    dataset, train classifiers and evalutes model
    with ten-fold cross validation.
    :param path: path of data-set
    :return: 10-fold accuracy of model
    """
    from os.path import join
    import copy as cp
    from sklearn import preprocessing
    from read_inputs import read_input_discrete
    from accuracy_of_model import ten_fold_cross_validation_without_scaling

    # read train
    pathTrain = join(path, path+'.arff')
    train = read_input_discrete(pathTrain)

    print 'Dataset: ', len(train['train'])

    TenFoldAcc = ten_fold_cross_validation_without_scaling(learningFunction=create_classifiers,
                                                           predictFuncion=predict, dataSet=train,
                                                           additionalArg=None)

    return TenFoldAcc

if '__main__' == __name__:
    print BRM_NB_DISCRETE('medical')