##############################################
# Computer exercise 6-a
##############################################
def uniform_distribution(x_l, x_u, n):
    """
    This function draw n samples from uniform distribution
    uniform(x_l, x_u)
    :param x_l: lower bound of uniform distribution
    :param x_up: upper bound of uniform distribution
    :param n: number of samples that should be drawn
    :return: return a list which contains n samples
    """
    import numpy as np

    # Generate and return n different discrete uniform random sample
    return np.random.random_integers(x_l, x_u, n).tolist()

##############################################
# Computer exercise 6-b
##############################################
def choose_bounds_and_size():
    """
    This function chooses bounds of discrete uniform
    distribution and number of samples.
    -100 <=lower bound of uniform distribution < upper bound of uniform distribution <= 100

    :return: it returns a list that contains three number
    [lower bound of uniform distribution,
     upper bound of uniform distribution,
     number of samples]
    """
    import numpy as np

    #choose lower and upper bound of uniform distribution
    while True:
        x_l, x_u = np.random.random_integers(-100, 100, 2)
        if x_l > x_u:
            x_l, x_u = x_u, x_l
        if x_l != x_u:
            break

    # choose number of samples
    n = np.random.random_integers(1, 1000)

    #return lower bound, upper bound, size
    return [x_l, x_u, n]


##############################################
# Computer exercise 6-c, d, e
##############################################
def generate_samples(n):
    """
    This function gets an scalar input which determine
    number of samples and it uses choose_bounds_and_size()
    to determine uniform distribution bound and num of samples
    that it should draw from the distribution each time,
    In each iteration it generates some samples then it calculates
    mean of samples.
    :param n: number of iterations
    :return: return n samples
    """
    # Output
    out = []

    # Randomly determine lower and upper bound of
    # Uniform discrete function and the number of samples
    x_l, x_u, size = choose_bounds_and_size()

    # Do n iteration
    for i in range(n):

        # Draw samples from uniform distribution (x_l, x_u)
        # and add average of them to output list
        samples = uniform_distribution(x_l, x_u, size)
        out.append(sum(samples)/len(samples))

    # Return n samples
    return out


def plot_histogram_and_norm_for_samples(n):
    """
    This function uses generate_samples function to generate n samples,
    then it plots histogram of it, and finally it calculates an
    approximate normal distribution which fits sample.
    As output it plots histogram and normal distribution of data
    :param n: number of samples
    :return: none
    """
    import matplotlib.pyplot as plt
    import matplotlib.mlab as mlab
    from scipy.stats import norm

    # Generate samples
    samples = generate_samples(n)

    # Plot histogram
    nn, bins, patches = plt.hist(x=samples, bins='auto', normed=True, facecolor="khaki")

    # Find a good normal distribution for samples
    mu, covariance = norm.fit(samples)

    # Add approximated normal distribution
    y = mlab.normpdf(bins, mu, covariance)
    plt.plot(bins, y, 'g--', linewidth=2, alpha=0.30)

    # plot
    plt.xlabel('Sample')
    plt.ylabel('Probability')
    plt.title('Samples size = {0}, Mean is {1} and variance is {2:0.2f}'.format(n ,mu, covariance))
    plt.grid(True)
    plt.show()

##############################################
# Test Computer exercise 6
##############################################

# Computer exercise 6-a
#print( uniform_distribution(0, 5, 10))

# Computer exercise 6-b
#print(choose_bounds_and_size())


# Computer exercise 6-c,d,e
from multiprocessing import Process

proccesses = []
for i in [10000, 100000, 1000000]:
    proccesses.append(Process(target=plot_histogram_and_norm_for_samples, args=[i]))

for p in proccesses:
    p.start()
for p in proccesses:
    p.join()

