import numpy as np
import matplotlib.pyplot as plt

# Number of samples
n = 10000

# Different value of mean and standard deviation
# for two normal distributions
Means1 = [-30, 15, 2, 6, 0]
Means2 = [30, 5, 8, 12, 0]
Sigma1 = [3, 5, 8, 10, 1]
Sigma2 = [30, 5, 3, 2, 1]

# Do experiment with different values
for i in range(5):
    mu1 = Means1[i]
    sigma1 = Sigma1[i]

    mu2 = Means2[i]
    sigma2 = Sigma2[i]

    # Draw samples from distribution 1
    samples1 = np.random.normal(loc=mu1, scale=sigma1, size=n)

    # Draw samples from distribution 2
    samples2 = np.random.normal(loc=mu2, scale=sigma2, size=n)

    # Add each element of sample1 to corresponding element of sample2
    samples3 = samples1 + samples2

    # Mean of samples3
    mu3 = np.mean(samples3)

    # Var of samples3
    var3 = np.cov(samples3)

    plt.figure()
    # Plot histogram of sample form distribution N(mu1, sigma1^2)
    plt.hist(x=samples1, bins='auto', normed=True, alpha=0.75,
             label='Samples1: N({}, {})'.format(mu1, sigma1 ** 2))

    plt.hist(x=samples2, bins='auto', normed=True, alpha=0.75,
             label='Samples2: N({}, {})'.format(mu2, sigma2 ** 2))

    # Plot histogram of sample form samples1+samples2
    plt.hist(x=samples3, bins='auto', normed=True, alpha=0.8,
             label='Dist1.+Dist2: N({},{})'.format(np.round(mu3, 2), np.round(var3, 2)))

    plt.grid()
    plt.legend()

plt.show()
