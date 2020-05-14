from math import sqrt
import numpy as np
import matplotlib.pyplot as pl

# Number of sample
numSample = 100

# Mean of normal distribution
mean = -1

# Variances
variances = [0.25, 0.5, 1]

# Standard deviation of normal distribution
standardDeviations  = list( map(sqrt, variances))

# numpy.random.normal function is used to generate sample with
# Three different variances that are given.
# List generator is used for neater code.
samples = [ list( np.random.normal(loc=mean, scale=sd, size=numSample)) for sd in standardDeviations]

# Plot Histograms
pl.figure()
for i in range(3):
    pl.subplot(1,3,i+1)
    # hist is used to draw Histogram, bins are selecting automatically
    pl.hist(x=samples[i], bins='auto', range=(-5,4), normed=True)
    pl.xlabel('samples')
    pl.ylabel('Probability')
    pl.ylim(0,1)
    pl.title('Mean={0}, Variance={1}'.format(mean, variances[i]))

pl.show()
