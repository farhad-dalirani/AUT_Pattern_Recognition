def phi(x, xi, h):
    """
    It calculates window function which is given in problemset
    :param x: center of window
    :param xi: given point
    :param h: len of hypercube
    :return: value of window function fot xi
    """
    import numpy as np

    # Convert to matrix
    if type(x) is not np.matrix:
        xMat = np.matrix(x)
    else:
        xMat = np.copy(x)

    if type(xi) is not np.matrix:
      xiMat = np.matrix(xi)
    else:
        xMat = np.copy(xi)

    # Calculate window function
    result = np.exp((-1/(2*h*h))*(xMat-xiMat)*np.transpose(xMat-xiMat))

    return result.tolist()[0][0]



def p_x_given_wj(wj, x, h):
    """
    This function uses parzen windows to estimate value of p(x\wj)
    :param wj: samples form wj
    :param x: a point which we want estimate probability of it
    :param h: size of hypercube
    :return: p(x|wj)
    """
    # Calculate parzen estimate
    parzenForAllSample = 0
    for sample in wj:
        # value of parzen window of x from parzen windows
        # which its center is sample
        parzenForAllSample += phi(x=sample,xi=x,h=h)

    # Return estimated p(x|wj)
    return parzenForAllSample/(len(wj)*h*h*h)


def classify(dataset, x, h):
    """
    determine class of x by calculating p(x|wi)p(wi) for all i
    :param dataset: dataset which is a dictionary and contain
                        sample of all classes {'w1':,'w2':,'w3':}
    :param x: input point
    :param h: length of hypercube
    :return: name of predicted class
    """
    # count number of elements
    n = 0
    for key in dataset:
        n += len(dataset[key])

    # find class with highest p(w|x)
    maxPro = -1
    predict = ''
    for key in dataset:
        # Calculate p(x|key)
        pxGivenW =  p_x_given_wj(wj=dataset[key],x=x,h=h) * (len(dataset[key])/n)
        if pxGivenW > maxPro:
            maxPro = pxGivenW
            predict = key
    return predict

# Dataset
dataset = {
    'w1':[ [0.28, 1.31, -6.2],[0.07, 0.58, -0.78],[1.54, 2.01, -1.63],
           [-0.44,1.18,-4.32],[-0.81,0.21,5.73],[1.52,3.16,2.77],
           [2.20,2.42,-0.19],[0.91,1.94,6.21],[0.65,1.93,4.38],[-0.26,0.82,-0.96]],
    'w2':[ [0.011,1.03,-0.21],[1.27,1.28,0.08],[0.13,3.12,0.16],
           [-0.21,1.23,-0.11],[-2.18,1.39,-0.19],[0.34,1.96,-0.16],
           [-1.38,0.94,0.45],[-0.12,0.82,0.17],[-1.44,2.31,0.14],[0.26,1.94,0.08]],
    'w3':[ [1.36,2.17,0.14],[1.41,1.45,-0.38],[1.22,0.99,0.69],
           [2.46,2.19,1.31],[0.68,0.79,0.87],[2.51,3.22,1.35],
           [0.60,2.44,0.92],[0.64,0.13,0.97],[0.85,0.58,0.99],[0.66,0.51,0.88]]
}



inputX = [[0.50, 1.0, 0.0],[0.31,1.51,-0.50],[-0.3,0.44,-0.1]]

# PART A
h = 1
for x in inputX:
    print('h={}, \t x={}, \tpredict={}'.format(h,x,
                                classify(dataset=dataset,x=x,h=h)))

# PART B
h = 0.1
for x in inputX:
    print('h={}, \t x={}, \tpredict={}'.format(h,x,
                                classify(dataset=dataset,x=x,h=h)))