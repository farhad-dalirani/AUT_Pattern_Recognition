import numpy as np
import matplotlib.pylab as plt
import scipy.io
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

def subtract_mean(observationMatrix):
    """
    This function calculates mean of
    observation matrix  and subtract
    observation matrix by mean.
    :param observationMatrix: is a matrix numpy object
    :return: a new matrix which is obtained by subtracting
            observation matrix by its mean. output is an
            numpy matrix and mean itself.
            (new observation matrix, mean)
    """
    meanOfSample = np.sum(observationMatrix, axis=1) / np.shape(observationMatrix)[1]
    observationMatrixSubtracted = observationMatrix - meanOfSample

    return  (observationMatrixSubtracted, meanOfSample)


def calculate_covariace(matrixX):
    """
    This functions calculate covariance of input matrix
    :param matrixX: is an observation matrix which its mean
            was subtracted. it's as numpy matrix object
    :return: covariance of input matrix, it's as numpy matrix object
    """
    # Calculate covariave matrix
    covariance = (matrixX * np.transpose(matrixX)) / (np.shape(matrixX)[1]-1)

    return covariance


def eigen_value_vector(matrixX):
    """
    Use numpy package to calculate eigen values and vectors
    :param matrixX: input n * n matrix
    :return: a pair like (eigen values, eigen vectors)
    """
    import numpy as np

    # Calculate eigen values and vector
    eigenValueVector = np.linalg.eig(matrixX)

    return eigenValueVector


########################################################
# A
########################################################

# Read Train file. which are .mat files
# Read Train
mat = scipy.io.loadmat('Subset1YaleFaces.mat')
train = mat['X']  #training data
trainY = mat['Y'] #labels
train = np.matrix(train)
trainFaces = np.transpose(train) / 255.0

# Read Test
mat2 = scipy.io.loadmat('Subset2YaleFaces.mat')
train2 = mat2['X']  #training data
trainY2 = mat2['Y'] #labels
train2 = np.matrix(train2)
trainFaces2 = np.transpose(train2) / 255.0

# Read second Test
mat3 = scipy.io.loadmat('Subset3YaleFaces.mat')
train3 = mat3['X']  #training data
trainY3 = mat3['Y'] #labels
train3 = np.matrix(train3)
trainFaces3 = np.transpose(train3) / 255.0


#print(type(trainFaces))
#print('>>>', np.shape(trainFaces))
#print(trainFaces)

# Calculate mean of faces and subtract mean
trainFacesMinesMean, meanFace = subtract_mean(observationMatrix=trainFaces)
#s = np.split(trainFaces,[60,61],axis=1)[1].reshape((50,50), order='F')

# Display Mean Image
meanImg = meanFace.reshape((50, 50), order='F')
fig, ax = plt.subplots()
cax = plt.imshow(meanImg, cmap='gray')
cbar = fig.colorbar(cax)
plt.title('Mean Face')

# Calculate 1/(n-1) * (transpose(X) * x)
covFace = calculate_covariace(matrixX=np.transpose(trainFacesMinesMean))

# Calculate Eigen Values and Eigen Vectors of faces
eigenVal, eigenVecprime = eigen_value_vector(matrixX=covFace)


# calculate eigen vector of (1/n * A*transpose(A)) from
# eigen vector of (1/n * transpose(A) * A).
# it is done by multiplying A at eigen vectors of 1/n * transpose(A) * A
eigenVec = trainFacesMinesMean * eigenVecprime

# normalize to unit l2 norm
eigenVec = preprocessing.normalize(eigenVec, axis=0, norm='l2')

# sort eigen values and their corresponding eigen vectors
idx = eigenVal.argsort()[::-1]
eigenVal = eigenVal[idx]
eigenVecprime = eigenVecprime[:,idx]
eigenVec = eigenVec[:,idx]


print('Number of Eigen Values of faces:\n{}'.format(len(eigenVal)))
print('Eigen Values of faces:\n{}'.format(eigenVal))
print('Eigen Vectorprime of faces:\n{}'.format(eigenVecprime))
print('Eigen Vector of faces:\n{}'.format(eigenVec))


# Display first nine Eigen Face
fig = plt.figure()
fig.suptitle("First 9 Eigen Faces")
for index in range(9):
    eigenFace = np.split(eigenVec, [index, index+1], axis=1)[1]

    # Display Eigen Faces
    eigenFaceImg = eigenFace.reshape((50, 50), order='F')
    plt.subplot(331+ index)
    plt.imshow(eigenFaceImg, cmap='gray')

# choose different m number of eigen vectors and project a photo on PCA space
# Choose A Face
theFace = np.split(trainFaces, [6, 7], axis=1)[1]
theFaceDrawing = theFace.reshape((50,50), order='F')

# Draw orginal face
indexInSubPlot = 1
fig = plt.figure()
fig.suptitle("A Face & its reconstructions using different number of Eigen Faces\nEF = Eigen Face")
plt.subplot(2,5,0+indexInSubPlot)
plt.imshow(theFaceDrawing, cmap='gray')
plt.title('Original')


for m in [5, 10, 15, 20, 30, 35, 40, 50, 70]:
    indexInSubPlot += 1

    # Choose first m eigen face
    mEigenFace = np.split(eigenVec, [0, m], axis=1)[1]

    # Project face on first m PCA
    projectedFace = np.transpose(mEigenFace) * theFace
    # Reconstruct face
    reconstructFace = mEigenFace * projectedFace

    reconstructFaceForDrawing = reconstructFace.reshape((50, 50), order='F')


    # Display Constructed Faces
    plt.subplot(2, 5, 0 + indexInSubPlot)
    plt.imshow(reconstructFaceForDrawing, cmap='gray')
    plt.title('First {} EF'.format(m))



############################################
# B
############################################
# For different number of neighbour
for k in [1, 3, 5, 7, 9]:
    for m in [5, 7, 10, 15, 25, 35, 45, 60]:
        # Choose first m eigen face
        mEigenFace = np.split(eigenVec, [0, m], axis=1)[1]

        # Project train face on first m PCA
        projectedFaceTrain = np.transpose(mEigenFace) * trainFaces

        # Project test face on first m PCA
        projectedFaceTest = np.transpose(mEigenFace) * trainFaces2

        # KNN Classifier
        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh.fit(projectedFaceTrain.T.tolist(), trainY.T.tolist()[0])
        prediction = neigh.predict(projectedFaceTest.T.tolist())

        err = 0
        correctPrediction = trainY2.T.tolist()[0]
        for index, element in enumerate(prediction):
            if element != correctPrediction[index]:
                err += 1

        # Print error for knn with K=k and  projection that uses first m eigen vector
        print('k={}\t, First M EigenVector={}\t, Error= {}%'.format(k, m, round((err/len(prediction))*100,2)))


k=5
m=15

# Choose first m eigen face
mEigenFace = np.split(eigenVec, [0, m], axis=1)[1]

# Project train face on first m PCA
projectedFaceTrain = np.transpose(mEigenFace) * trainFaces

# Project test2(dataset yale3) face on first m PCA
projectedFaceTest2 = np.transpose(mEigenFace) * trainFaces3

# KNN Classifier
neigh = KNeighborsClassifier(n_neighbors=k)
neigh.fit(projectedFaceTrain.T.tolist(), trainY.T.tolist()[0])
prediction = neigh.predict(projectedFaceTest2.T.tolist())

err = 0
correctPrediction = trainY3.T.tolist()[0]
for index, element in enumerate(prediction):
    if element != correctPrediction[index]:
        err += 1

print("Error of Dataset Yale3:")
print('k={}\t, First M EigenVector={}\t, Error= {}%'.format(k, m, round((err/len(prediction))*100,2)))


plt.show()
