# Continues Input
# Independent classifiers
# No parent
# Base Classifier: SVM
# Training Scheme: -
# Select Root: -
# No Chain


def create_classifiers(train):
    """
    This function creates classifiers for each class.
    This is an binary relevance method, so classifiers are
    independent. SVM is the base classier.

    :param train: training dataset
    :param parents: a dictionary that contains parent of each node
    :param topologicalSort: a topological sort of classifiers
    :return: a dictionary trained classifiers
    """
    from sklearn.svm import SVC
    from read_inputs import create_training_set_for_binary_relevance_classifier

    # A dictionary of classifiers for each class
    # {class0: classifier0, ...,classN: classifierN}
    classifiers = {classifier: None for classifier in range(0, train['numberOfLabels'])}

    # for each class makes an gaussian naive bayes classifiers and train it
    for key in classifiers:
        classifiers[key] = SVC()

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


def BRM_SVM(path):
    """
        This function uses other functions to read
        dataset, train classifiers and evalutes model
        with ten-fold cross validation.
        :param path: path of data-set
        :return: 10-fold accuracy of model
    """
    from os.path import join
    from read_inputs import read_input_continues
    from accuracy_of_model import ten_fold_cross_validation_with_scaling

    # read dataset
    pathTrain = join(path, path+'.arff')
    train = read_input_continues(pathTrain)

    print 'Dataset: ', len(train['train']), ' ', train

    TenFoldAcc = ten_fold_cross_validation_with_scaling(learningFunction=create_classifiers,
                                                        predictFuncion=predict, dataSet=train,
                                                        additionalArg=None)

    return TenFoldAcc


if '__main__' == __name__:
    print BRM_SVM('yeast')