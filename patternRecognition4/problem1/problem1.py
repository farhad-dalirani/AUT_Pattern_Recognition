import numpy as np
import matplotlib.pylab as plt

# Dataset
dataset = {'sunny': [4, 1, 5], 'cloudy': [3, 2]}

n = 5
h = 1

# Plotting p(x|w1)
x_i = np.linspace(start=-1, stop=8, num=400)

# p(xi|w1) for xi in x_i
px_i = []

# Calculate p(x_i|w1) for each element of x_i
for element in x_i:
    total_phi = 0
    # For different point of w1(sunny)
    for x in dataset['sunny']:
        # Calculate phi((x-xi)/h)
        if np.abs((x-element)/h) < 0.5:
            # point is inside window
            total_phi += 1
    # Add p(xi|w1)
    px_i.append(total_phi/(len(dataset['sunny'])*h))

# Plot x_i, p(x_i|W1)
plt.plot(x_i, px_i)
# Plot sunny days
plt.plot(dataset['sunny'], [0]*len(dataset['sunny']), 'r*')
plt.xlabel('x_i')
plt.ylabel('p(x_i|w1)')
plt.grid()
plt.show()


# Plotting p(x|w2)
# p(xi|w2) for xi in x_i
px_i = []

# Calculate p(x_i|w2) for each element of x_i
for element in x_i:
    total_phi = 0
    # For different point of w1(sunny)
    for x in dataset['cloudy']:
        # Calculate phi((x-xi)/h)
        if np.abs((x-element)/h) < 0.5:
            # point is inside window
            total_phi += 1
    # Add p(xi|w1)
    px_i.append(total_phi/(len(dataset['cloudy'])*h))

# Plot x_i, p(x_i|W1)
plt.plot(x_i, px_i)
# Plot sunny days
plt.plot(dataset['cloudy'], [0]*len(dataset['cloudy']), 'r*')
plt.xlabel('x_i')
plt.ylabel('p(x_i|w2)')
plt.grid()
plt.show()
