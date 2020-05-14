import numpy as np
import matplotlib.pylab as plt


def px_given_theta(x, theta):
    """
    Calculate p(x|theta) of problem3-Q1
    :param x: a list of point
    :param theta: a parameter of distribution
    :return: value of P(x|theta) at points of x
    """
    result = []
    for point in x:
        if point < 0:
            result.append(0)
        else:
            result.append(theta * np.exp(-1*theta*point))

    return result


############################################################
# A and C
############################################################
# Generate 100 point between 0 and 5
x = np.linspace(start=0, stop=5, num=10000)

# Plot p(x|theta=1)
plt.plot(x, px_given_theta(theta=1, x=x))

# draw 100000 sample randomly from exponential distribution
# with theta 1
samples = np.random.exponential(scale=1, size=100000)

# Calculate optimal theta by maximum likelihood
thetaMLE = 1/((1/len(samples)) * sum(samples))
thetaMLE = round(thetaMLE)

# Mark theta optimal
plt.plot(thetaMLE, 0, 'r*', label="Theta' obtained by MLE= {}".format(thetaMLE))

plt.grid()
plt.legend()
plt.xlabel('x')
plt.ylabel('P(x|theta)')
plt.title('p(x|theta),   theta=1')
plt.show()


# Generate 100 point between 0 and 5
theta = np.linspace(start=0, stop=5, num=10000)

# Calculate corresponding p(x|theta)
px_theta = [px_given_theta(x=[2], theta=t) for t in theta]

# Plot p(x|theta) for x=2 and theta in [0,5]
plt.plot(theta, px_theta)
plt.grid()
plt.ylabel('P(x|theta)')
plt.title('p(x|theta), 0 <= Theta <= 5, x=2')
plt.show()

