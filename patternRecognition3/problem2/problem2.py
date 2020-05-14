import matplotlib.pylab as plt
import numpy as np


def likelihood_uniform(theta, numberOfSample, x):
    """
    Likelihood function of uniform distribution
    uniform(0, theta) when we have n sample
    X1, X2,..., Xn is:
    L(Theta) = (1/Theta)^ number of samples
    :param theta: determine interval of uniform
            distribution U(0, theta)
    :param x: a list of points which we
            want to calculate
            likelihood function at those point
    :return: values of likelihood at x
    """
    result = []
    for point in x:
        if point < theta:
            result.append(0)
        else:
            result.append((1/point)**numberOfSample)
    return result


# Generate 100 point between 0 and 2
x = np.linspace(start=0, stop=2, num=100)

# Plot likelihood
plt.plot(x, likelihood_uniform(theta=0.6, numberOfSample=5, x=x))
plt.grid()
plt.xlabel('Theta')
plt.ylabel('Likelihood (1/Theta)^N')
plt.show()