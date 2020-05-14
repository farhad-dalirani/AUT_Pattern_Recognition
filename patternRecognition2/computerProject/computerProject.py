import numpy as np
import matplotlib.pyplot as plt


########################################################################
# Computer exercise 7-a
########################################################################


def d_normal_distribution(mu, covMat, n):
    """
    D-dimensional normal distribution function,
       it draws samples from a D-dimensional normal distribution.
     :param
       (mu)mean vector,
      (covMat)covariance matrix,
      (n)number of samples that should be drawn.
     :return Output:
       each entry out[i,j,...,:] is an D-dimensional value drawn from normal distribution.
    An example of this function:
        d_normal_distribution(mu=[0,0],covMat=[[1, 0],[0, 100]], n=10)
    """
    import numpy as np
    # Use multivariate normal distribution of numpy package
    return np.random.multivariate_normal(mean=mu, cov=covMat, size=n, check_valid='ignore').tolist()


########################################################################
# Computer exercise 7-b
########################################################################

# Distributions of classes
xw1 = {'means': [1, 0], 'covriance': [[1, 0], [0, 1]], 'prior': 0.5, 'name': 'w1'}
xw2 = {'means': [-1, 0], 'covriance': [[1, 0], [0, 1]], 'prior': 0.5, 'name': 'w2'}

# As I mathematically have shown in report, decision boundary is x0 = 0.
# Plot decision boundary
y = np.linspace(-5, 5, 100)
plt.plot([0]*100 , y, 'b--', label='Decision Boundary X0 = 0')
plt.title("Decision Boundary")
plt.grid()
plt.legend()
plt.show()


########################################################################
# Computer exercise 7- c, d
########################################################################
def bhattacharyya_bound(normalDist1, normalDist2):
    """
    This measure shows similarities between two
    distribution.
    :param normalDist1: is dictionary that contains
            these keys {'means','covariance', 'name','prior'}
    :param normalDist2: is dictionary that contains
            these keys {'means','covariance', 'name','prior'}
    :return: return a scalar
    """
    import numpy as np

    # Convert data to proper form
    mean1 = np.asmatrix(normalDist1['means'])
    mean2 = np.asmatrix(normalDist2['means'])
    cov1 = np.asmatrix(normalDist1['covriance'])
    cov2 = np.asmatrix(normalDist2['covriance'])

    # Average of two covariances
    covAve = (cov1 + cov2) / 2

    # Subtract of means
    meanSub = mean2 - mean1

    # Calculate the bound
    bound = (np.sqrt(normalDist1['prior'] * normalDist2['prior'])) * (
            np.exp(-1 * ((1/8) * (meanSub * np.linalg.inv(covAve) * np.transpose(meanSub)) +
                         (1/2) * np.log(
                             np.linalg.det(covAve)/np.sqrt(np.linalg.det(cov1) *
                                                           np.linalg.det(cov2))
                                         ))))
    #return bound
    return bound.tolist()[0][0]


# Print Bhattacharyya bound
bhb = bhattacharyya_bound(xw1, xw2)
print('Bhattacharyya bound for w1 & w2:{:f}'.format(bhb))

# For plotting Errors by increasing number of samples
errors = []

# Generate samples for both distribution and classify them
# and calculate empirical error of them
# Do it for different number of samples 100, 200, 300, ..., 1000
startNumOfSamples = 50
endNumOfSamples = 501
stepOfSample = 50
for numOfSamples in np.arange(startNumOfSamples, endNumOfSamples, stepOfSample):
    # Draw samples from distribution 1
    w1 = d_normal_distribution(mu=xw1['means'], covMat=xw1['covriance'],
                               n=numOfSamples)

    # Draw samples from distribution 2
    w2 = d_normal_distribution(mu=xw2['means'], covMat=xw2['covriance'],
                               n=numOfSamples)

    error = 0
    for samples in w1:
        # Decision Boundary: x0 = 0
        if(samples[0] < 0):
            error += 1

    for samples in w2:
        # Decision Boundary: x0 = 0
        if (samples[0] > 0):
            error += 1

    print("Number of sample {}, Error is {}".format(numOfSamples*2, round(error/(len(w1)+len(w2)),3)))

    # Add error to the list of error
    errors.append(round(error/(len(w1)+len(w2)),3))

plt.plot(np.arange(startNumOfSamples*2,
                   endNumOfSamples*2-1,
                   stepOfSample*2).tolist(), errors, 'r-', label="Errors")

plt.plot(np.arange(startNumOfSamples*2,
                   endNumOfSamples*2-1,
                   stepOfSample*2).tolist(), [bhb]*len(errors),'b-', label="Bhattacharyya bound")
plt.legend()
plt.xlabel("Number Of Samples")
plt.show()



