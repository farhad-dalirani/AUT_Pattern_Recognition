import numpy as np
import matplotlib.pyplot as pl
from matplotlib.colors import LogNorm

# Number of samples
numSample = 10000

# Number of Bins
numBin = 40

# Mean of normal distribution
means = ((1,1), (-1,1), (1,0))

# Covariance Matrix
Covariances = (((2, 0), (0, 2)), ((2, 0), (0, 5)), ((2, 5), (5, 3)))

# For each set of mean and covariance,
# generate sample and draw histogram
for mu, co in zip(means, Covariances):

    # Generate Normal Distribution samples for each set of mean and covariance
    samples = np.random.multivariate_normal(mean=mu, cov=co, size=numSample, check_valid="warn")

    # Separate x and y of samples
    samplesX = []
    samplesY = []
    for sample in samples:
        samplesX.append(sample[0])
        samplesY.append(sample[1])

    # Plot Histogram,
    pl.figure()

    # LogNorm is used to make colors distinguishable
    pl.hist2d(samplesX, samplesY, bins=numBin, norm=LogNorm())

    pl.title("mean={0}, Covariance={1}".format(mu,co))
    # colorbar is used to specify probability of a bar
    pl.colorbar()
    pl.xlabel("X")
    pl.ylabel("Y")
    pl.grid()

pl.show()
