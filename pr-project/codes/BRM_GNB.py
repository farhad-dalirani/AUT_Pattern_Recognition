#########################################################
#   This Code Isn't In The Paper, I've Implemented It   #
#   For Seeing Results Of Different Methods.            #
#                                                       #
#   Binary Relevance Naive Bayes Is In File:            #
#   BRM_NB_DISCRETE.py                                  #
#                                                       #
#########################################################


# Continues Input
# Independent classifiers
# No parent
# Base Classifier: Naive Bayes
# Training Scheme: -
# Select Root: -
# No Chain


def create_classifiers(train):
    """
    create classifiers for each class and link them according to
    directed tree which is built from Chow & Liu's algorithm(MWST)
    :param train: training dataset
    :param parents: a dictionary that contains parent of each node
    :param topologicalSort: a topological sort of classifiers
    :return: a dictionary trained classifiers
    """
    from sklearn.naive_bayes import GaussianNB
    from read_inputs import create_training_set_for_binary_relevance_classifier

    # A dictionary of classifiers for each class
    # {class0: classifier0, ...,classN: classifierN}
    classifiers = {classifier: None for classifier in range(0, train['numberOfLabels'])}

    # for each class makes an gaussian naive bayes classifiers and train it
    for key in classifiers:
        classifiers[key] = GaussianNB()

        # Create appropriate classifier
        trainingSet = create_training_set_for_binary_relevance_classifier(train=train,
                                                         classOfClassifier=key)

        # Train classifier
        classifiers[key].fit(trainingSet[0], trainingSet[1])

    return classifiers


def predict(classifiers, testSample):
    """
    Use classifiers according their order to predict label of
    test samples.
    :param classifiers: a dictionary of trained classifiers
    :param testSamples: A test sample which this function
            wants to predict its labels
    :return: list of labels
    """
    # a dictionary for holding prediction for classes
    # it supposed 0 and -1 [maybe probability * should be checked]
    predictions = {Class: None for Class in classifiers}

    for _class in classifiers:

        label = classifiers[_class].predict([testSample])
        predictions[_class] = label[0]

    return predictions


def BRM_GNB(path):

    from os.path import join
    import copy as cp
    from read_inputs import read_input_continues
    from accuracy_of_model import ten_fold_cross_validation_with_scaling

    # read and scale train
    pathTrain = join(path, path+'.arff')
    train = read_input_continues(pathTrain)

    print 'Dataset: ', len(train['train']), ' ', train

    TenFoldAcc = ten_fold_cross_validation_with_scaling(learningFunction=create_classifiers,
                                                        predictFuncion=predict, dataSet=train,
                                                        additionalArg=None)

    return TenFoldAcc


if '__main__' == __name__:
    print BRM_GNB('emotions')