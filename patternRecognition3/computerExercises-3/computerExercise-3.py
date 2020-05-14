w3 = [[0.83, 1.6, -0.014],
[1.1, 1.6, 0.48],
[-0.44, -0.41, 0.32],
[0.047, -0.45, 1.4],
[0.28, 0.35, 3.1],
[-0.39, -0.48, 0.11],
[0.34, -0.079, 0.14],
[-0.3, -0.22, 2.2],
[1.1, 1.2, -0.46],
[0.18, -0.11, -0.49]]
###############################################################
# Computer Exercise 3 - needed function
###############################################################
def MLE_var(observationMatrix, mu):
    """
    Calculate variance of input observation matrix by MLE
    MLE covariance = 1/n * ((X1-mu)(X1-mu)T+(X2-mu)(X2-mu)T+...+(Xn-mu)(Xn-mu)T)
            , Xi is an observation
    in dataset
    :param observationMatrix:
    :param mu: mean of observations which is obtained by MLE_mu
    :return: estimated variance
    """
    import numpy as np

    # Convert to numpy matrix for matrix operation
    observationMatrix_np= np.matrix(observationMatrix)

    # Convert to numpy matrix
    mu_np = np.matrix(mu)

    # inital sum of (Xi-mu)(Xi-mu)T
    sumOfdiff = np.array([0]*len(observationMatrix_np[0]))

    # Calculate sum of (Xi-mu)(Xi-mu)T
    for observaton in observationMatrix_np:
        sumOfdiff = sumOfdiff + \
                    (np.transpose(observaton-mu_np)*(observaton-mu_np))

    # divide by number of observation
    sumOfdiff /= len(observationMatrix)

    # return mean
    return sumOfdiff

def MLE_mu(observationMatrix):
    """
    Calculate mean of input observation matrix by MLE
    MLE mu = 1/n * (X1+X2+...+Xn) , Xi is an observation
    in dataset
    :param observationMatrix:
    :return: estimated mu
    """
    import numpy as np

    # Convert to numpy array for matrix operation
    observationMatrix_np= np.array(observationMatrix)

    # inital sum of observations
    sumOfObservation = np.array([0]*len(observationMatrix_np[0]))

    # Calculate sum of observations
    for observaton in observationMatrix_np:
        sumOfObservation = sumOfObservation + observaton

    # divide by number of observation
    sumOfObservation /= len(observationMatrix)

    # return mean
    return sumOfObservation.tolist()


def normalDist(mu, var, x):
    """
    Calculate value of normal distribution at x
    :param mu: mean of normal distribution
    :param var: variance of normal distribution
    :param x: a plist of points
    :return: value of points of x in normal distribution
    """
    import numpy as np
    res = []
    for point in x:
        res.append((1/np.sqrt(2*np.pi*var)) * np.exp(((point-mu)**2)/(-2*var)))
    return res


###############################################################
# Computer Exercise 3 - a
###############################################################
def p_x_given_D(mu0, sigma0, sigma, D):
    """
    Plot p(x|D)~N(muN, sigma+sigmaN), muN and sigmaN are
    calculated by using mu0, sigma0, sigma and D
    :param mu0: p(mu) is a NormalDensity(mu0, sigma0)
    :param sigma0: p(mu) is a NormalDensity(mu0, sigma0)
    :param sigma: p(x|mu) is a normalDensity(mu, sigma)
    :param D: dataset
    :return: none
    """
    import numpy as np
    import matplotlib.pylab as plt

    D = [d[0] for d in D]

    # Calculate sigmaN^2
    sigmaN_pow2 = (sigma0**2 * sigma**2)/((len(D)*sigma0**2)+sigma**2)

    # Calculate muN
    muN = ((len(D)*sigma0**2)/(len(D)*sigma0**2 + sigma**2))*(sum(D)/len(D)) +\
          (((sigma**2)/(len(D)*(sigma0**2)+sigma**2)))*mu0

    # Generate 100 point around muN
    x = np.linspace(start=muN-4, stop=muN+4, num=100)

    # Generate points of p(x|D)~N(mu0, sigma+sigma0)
    px_D = normalDist(mu=muN, var=sigma**2 + sigmaN_pow2, x=x)

    # plot p(x|D)
    plt.plot(x, px_D, 'g-', label='P(x|D)')

    # plot dataset
    #datasetInNormalDensity = normalDist(mu=muN, var=sigma**2 + sigmaN_pow2, x=D)
    #plt.plot(x, px_D, 'b*', label='Dataset')

    plt.legend()
    plt.grid()
    plt.xlabel('X')
    plt.ylabel('P(x|D)')
    plt.title("p(x|D), Dogmatism: {}, muN: {}, SigmaN^2: {},\nmu0: {}, Sigma0^2: {}, Sigma^2:{}".format(
        np.round(sigma**2/sigma0**2, 2),
        np.round(muN, 2),
        np.round(sigmaN_pow2, 2),
        np.round(mu0, 2),
        np.round(sigma0**2, 2),
        np.round(sigma**2, 2)))
    plt.show()


###############################################################
# Computer Exercise 3 - b
###############################################################
# Calculate mean and variance for each feature of w1
import numpy as np

w3_f2 = []
for observation in w3:
    w3_f2.append(observation[2:3])

meanf2 = MLE_mu(w3_f2)

varf2 = MLE_var(w3_f2, meanf2)

# Estimationg sigma for x2 of w3
sigma = varf2.tolist()[0][0]
sigma = np.sqrt(sigma)

print("part-B)")
print('Estimated Mean for x2 of w3:\n {}'.format(meanf2[0]))
print('Estimated Sigma for x2 of w3:\n {}'.format(sigma))
print('Estimated variance(Sigma^2) for x2 of w3:\n {}'.format(sigma**2))

for dogmatism in [0.1, 1.0, 10, 100]:
    sigma0 = np.sqrt(sigma**2 / dogmatism)
    p_x_given_D(mu0=-1, sigma0=sigma0, sigma=sigma, D=w3_f2)