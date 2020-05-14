import numpy as np
import matplotlib.pylab as plt


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


#############################################
# Data
#############################################
mu1 = [10, 10]
mu2 = [22, 10]

cov1 = [[4, 4], [4, 9]]
cov2 = [[4, 4], [4, 9]]

# Generate 1000 sample for class 1
samples1 = np.random.multivariate_normal(mean=mu1, cov=cov1, size=1000)
# Generate 1000 sample for class 1
samples2 = np.random.multivariate_normal(mean=mu2, cov=cov2, size=1000)


samples1 = np.matrix(samples1).T
samples2 = np.matrix(samples2).T

# All samples
allsamples = np.concatenate((samples1,samples2), axis=1)
allsamples = np.matrix(allsamples)
print('Dataset class1:\n', samples1)
print('Dataset class2:\n', samples2)
print('Dataset:\n', allsamples)
#############################################
# A
#############################################
# Caluculate PCAs
# Subtract mean
observationSubtractedMean, means = subtract_mean(observationMatrix=allsamples)

# Calculate Covariance Matrix
covOfObservation = calculate_covariace(matrixX=observationSubtractedMean)
print("Covariance For PCA:\n {}".format(covOfObservation))

# Calculate EigenVectors and EigenValues
eigenValues, eigenVectors = eigen_value_vector(matrixX=covOfObservation)
print("Eigen Values for covariance:\n {}".format(eigenValues))
print("Eigen Vectors for covariance:\n {}".format(eigenVectors))

# choose eigen vector with highest eigenvalue
if eigenValues[0] > eigenValues[1]:
    pcaVector = np.split(eigenVectors, [1], axis=1)[0]
    pcaVal = eigenValues[0]
else:
    pcaVector = np.split(eigenVectors, [1], axis=1)[1]
    pcaVal = eigenValues[1]

print("Eigen Vector with highest Eigen Value:\n {}".format(pcaVector))

# Plot  Data
plt.plot(samples1[0].tolist()[0], samples1[1].tolist()[0], 'r*',
         label='sample1, mu:{}'.format(mu1), alpha=0.5)
plt.plot(samples2[0].tolist()[0], samples2[1].tolist()[0], 'b*',
         label='sample2, mu:{}'.format(mu2), alpha=0.5)
# Plot PCA
plt.arrow(means.item((0,0)), means.item((1,0)),
          pcaVector.item(0)*pcaVal/40,
          pcaVector.item(1)*pcaVal/40,
          0.5, linewidth=1, head_width=1, color='green')
x0 = np.linspace(start=0, stop=30, num=100)
plt.plot(x0,
         [(x-means.item((0,0)))*(pcaVector.item(1)/pcaVector.item(0))+means.item((1,0)) for x in x0],
         'g--', label='PCA')

plt.legend()
#############################################
# B
#############################################
# project data on PCA
samples1Projection = np.transpose(pcaVector) * samples1
samples2Projection = np.transpose(pcaVector) * samples2

#plot projected data on pca
# Plot  Data
f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
plt.title('Projection On PCA')
ax1.plot(samples1Projection, [0]*len(samples1Projection), 'ro',
         label='sample1, mu:{}'.format(mu1), alpha=0.5)
ax2.plot(samples2Projection, [0]*len(samples2Projection), 'b^',
         label='sample2, mu:{}'.format(mu2), alpha=0.5)
ax3.plot(samples1Projection, [0]*len(samples1Projection), 'ro',
         label='sample1, mu:{}'.format(mu1), alpha=0.5)
ax3.plot(samples2Projection, [0]*len(samples2Projection), 'b^',
         label='sample2, mu:{}'.format(mu2), alpha=0.5)
f.subplots_adjust(hspace=0)
plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)


#############################################
# D
#############################################
samples1Reconstruct = pcaVector * samples1Projection
samples2Reconstruct = pcaVector * samples2Projection
allsamplesReconstruct = np.concatenate((samples1Reconstruct,samples2Reconstruct), axis=1)

# Calculate reconstruction error
# For that calculate sum of distance of
# points to their corresponding reconstructions
reconstructionError = 0
for index in range(np.shape(allsamplesReconstruct)[1]):
    sample = np.split(allsamples, [index, index+1], axis=1)[1]
    sampleReconstructed = np.split(allsamplesReconstruct, [index, index+1], axis=1)[1]
    diff = sample-sampleReconstructed

    reconstructionError += np.sqrt(diff.item(0, 0)**2+diff.item(1, 0)**2)


print('Reconstruction Error is {}'.format(reconstructionError))

# Plot  Reconstructed Data
plt.figure()
plt.plot(samples1Reconstruct[0].tolist()[0], samples1Reconstruct[1].tolist()[0], 'r*',
         label='sample1, mu:{}'.format(mu1), alpha=0.5)
plt.plot(samples2Reconstruct[0].tolist()[0], samples2Reconstruct[1].tolist()[0], 'b*',
         label='sample2, mu:{}'.format(mu2), alpha=0.5)
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim(0, 35)
plt.ylim(-10, 10)
plt.title('Reconstruction Error is {}'.format(reconstructionError))

#####################################################
# E
#####################################################
# Mean of x1
mean1 = np.sum(samples1, axis=1) / np.shape(samples1)[1]
samples1MinesMean1 = samples1 - mean1
print('Mean Class 1:\n {} \n'.format(mean1))

# Mean of x2
mean2 = np.sum(samples2, axis=1) / np.shape(samples2)[1]
samples2MinesMean2 = samples2 - mean2
print('Mean Class 2:\n {} \n'.format(mean2))

# mean of all elements
meanTotal = (np.sum(samples1, axis=1)+np.sum(samples2, axis=1))/\
            (np.shape(samples1)[1]+np.shape(samples2)[1])

print('Mean of all elements:\n {} \n'.format(mean2))

# Calculate scatter 1: s1 = cov1 * (n1 - 1)
s1 = np.cov(samples1MinesMean1) * (np.shape(samples1MinesMean1)[1]-1)
print('Scatter Class 1:\n {} \n'.format(s1))

# Calculate scatter 1: s1 = cov1 * (n1 - 1)
s2 = np.cov(samples2MinesMean2) * (np.shape(samples2MinesMean2)[1]-1)
print('Scatter Class 2:\n {} \n'.format(s2))

# Calculate S within
Sw = s1 + s2
print('Scatter within(Sw):\n {} \n'.format(Sw))

# Calculate W = inverse sw * (mean1 - mean2)
w = np.linalg.inv(Sw) * (mean1 - mean2)
print('W:\n {} \n'.format(w))

# Normalize w
w = w / np.sqrt(w.item((0, 0))**2 + w.item((1, 0))**2)


# Plot  Data
plt.figure()
plt.plot(samples1[0].tolist()[0], samples1[1].tolist()[0], 'r*',
         label='sample1, mu:{}'.format(mu1), alpha=0.5)
plt.plot(samples2[0].tolist()[0], samples2[1].tolist()[0], 'b*',
         label='sample2, mu:{}'.format(mu2), alpha=0.5)
# Plot PCA
plt.arrow(means.item((0,0)), means.item((1,0)),
          w.item(0)*2,
          w.item(1)*2,
          0.5, linewidth=1, head_width=1, color='green')
x0 = np.linspace(start=0, stop=30, num=100)
plt.plot(x0,
         [(x-meanTotal.item((0,0)))*(w.item(1)/w.item(0))+meanTotal.item((1,0)) for x in x0],
         'g--', label='LDA')

plt.legend()
plt.title('Dataset And LDA')


#####################################################
# F
#####################################################
# project data on PCA
samples1ProjectionOnLDA = np.transpose(w) * samples1
samples2ProjectionOnLDA = np.transpose(w) * samples2

#plot projected data on pca
# Plot  Data
f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
plt.title('Projection On LDA')
ax1.plot(samples1ProjectionOnLDA, [0]*len(samples1ProjectionOnLDA), 'ro',
         label='sample1, mu:{}'.format(mu1), alpha=0.5)
ax2.plot(samples2ProjectionOnLDA, [0]*len(samples2ProjectionOnLDA), 'b^',
         label='sample2, mu:{}'.format(mu2), alpha=0.5)
ax3.plot(samples1ProjectionOnLDA, [0]*len(samples1ProjectionOnLDA), 'ro',
         label='sample1, mu:{}'.format(mu1), alpha=0.5)
ax3.plot(samples2ProjectionOnLDA, [0]*len(samples2ProjectionOnLDA), 'b^',
         label='sample2, mu:{}'.format(mu2), alpha=0.5)
f.subplots_adjust(hspace=0)
plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)

plt.show()










