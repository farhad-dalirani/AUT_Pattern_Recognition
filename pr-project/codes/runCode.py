from BRM_NB_DISCRETE import BRM_NB_DISCRETE
from tnbcc_DISCRETE import tnbcc_DISCRETE
from BRM_SVM import BRM_SVM
from tsvmcc import tsvmcc

# DataSets that their features are discrete
#discreteDatasets = ['delicious', 'enron', 'medical', 'bibtex']
discreteDatasets = ['enron', 'medical']

# DataSets that their features are continuous
continuousDatasets = ['emotions', 'scene', 'yeast']

accuraciesDiscrete = {}
# Apply Binary Relevance Naive Bayes and
# Tree Naive Bayes Chain Classifier
for dataSet in discreteDatasets:
    accuraciesDiscrete[dataSet] = {'BRM_NB_DISCRETE': None, 'tnbbcc_DISCRETE': None}

    # Binary Relevance Naive Bayes
    accuraciesDiscrete[dataSet]['BRM_NB_DISCRETE'] = BRM_NB_DISCRETE(dataSet)

    # Tree Naive Bayes Chain Classifier
    accuraciesDiscrete[dataSet]['tnbbcc_DISCRETE'] = tnbcc_DISCRETE(dataSet)


accuraciesContinuous = {}
# Apply Binary Relevance SVM and
# Tree SVM Chain Classifier
for dataSet in continuousDatasets:
    accuraciesContinuous[dataSet] = {'BRM_SVM': None, 'tsvmbcc': None}

    # Binary Relevance SVM
    accuraciesContinuous[dataSet]['BRM_SVM'] = BRM_SVM(dataSet)

    # Tree SVM Chain Classifier
    accuraciesContinuous[dataSet]['tsvmbcc'] = tsvmcc(dataSet)

print '=========================================='
for dataSetAcc in accuraciesDiscrete:
    print 'Dataset Name:', dataSetAcc, '\n'
    print accuraciesDiscrete[dataSetAcc], '\n'

print '=========================================='
for dataSetAcc in accuraciesContinuous:
    print 'Dataset Name:', dataSetAcc, '\n'
    print accuraciesContinuous[dataSetAcc], '\n'
