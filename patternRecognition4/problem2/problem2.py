import matplotlib.pylab as plt

def draw_decision_boundary_of_knn(wi, wj, wk, title):
    """
    This function gets points of two class wi and wj
    and calculates decision boundary of 1-nearest neighbour
    :param wi: a list of points which belong to class wi
    :param wj: a list of points which belong to class wj
    :return: none
    """
    import pylab as pl
    import numpy as np

    # calculate means
    if wk != None:

        #mean of wi
        meanwi = [0,0]
        for point in wi:
            meanwi[0] += point[0]
            meanwi[1] += point[1]
        meanwi[0] /= len(wi)
        meanwi[1] /= len(wi)

        # mean of wj
        meanwj = [0, 0]
        for point in wj:
            meanwj[0] += point[0]
            meanwj[1] += point[1]
        meanwj[0] /= len(wj)
        meanwj[1] /= len(wj)

        # mean of wk
        meanwk = [0, 0]
        for point in wk:
            meanwk[0] += point[0]
            meanwk[1] += point[1]
        meanwk[0] /= len(wk)
        meanwk[1] /= len(wk)

    else:

        # mean of wi
        meanwi = [0, 0]
        for point in wi:
            meanwi[0] += point[0]
            meanwi[1] += point[1]
        meanwi[0] /= len(wi)
        meanwi[1] /= len(wi)

        # mean of wj
        meanwj = [0, 0]
        for point in wj:
            meanwj[0] += point[0]
            meanwj[1] += point[1]
        meanwj[0] /= len(wj)
        meanwj[1] /= len(wj)


    # This determines how many points make the plot.
    # Number of points on plot: plot_resolution * plot_resolution
    plot_resolution = 400

    # Make points of plot
    X, Y = np.mgrid[-11:11:plot_resolution*1j, -11:11:plot_resolution*1j]

    # Concatenate X,Y
    points = np.c_[X.ravel(), Y.ravel()]

    # class of points
    pointClass = np.array([0]*len(points))

    # determine the nearest of each point of plan which we are
    # drawing
    for index, point in enumerate(points):

        # find samples which has minimum distance to point
        minDis = np.inf
        classOfPoint = -1

        # compare distance to samples of class wi
        for sample in wi:
            # distance of point to a sample of wi
            distance = np.sqrt((point[0]-sample[0])**2+(point[1]-sample[1])**2)
            if distance < minDis:
                minDis = distance
                classOfPoint = 1

        # compare distance to samples of class wj
        for sample in wj:
            # distance of point to a sample of wj
            distance = np.sqrt((point[0] - sample[0]) ** 2 + (point[1] - sample[1]) ** 2)
            if distance < minDis:
                minDis = distance
                classOfPoint = 2

        if wk != None:
            # compare distance to samples of class wk
            for sample in wk:
                # distance of point to a sample of wk
                distance = np.sqrt((point[0] - sample[0]) ** 2 + (point[1] - sample[1]) ** 2)
                if distance < minDis:
                    minDis = distance
                    classOfPoint = 3

        # distance of point to means of different classes
        dismean1 = np.sqrt((point[0] - meanwi[0]) ** 2 + (point[1] - meanwi[1]) ** 2)
        dismean2 = np.sqrt((point[0] - meanwj[0]) ** 2 + (point[1] - meanwj[1]) ** 2)
        if wk != None:
            dismean3 = np.sqrt((point[0] - meanwk[0]) ** 2 + (point[1] - meanwk[1]) ** 2)

        # Decision boundary between means
        if np.abs(dismean1 - dismean2) > 0 and np.abs(dismean1 - dismean2) < 0.01:
            classOfPoint = 0
        if wk != None:
            if np.abs(dismean1 - dismean3) > 0 and np.abs(dismean1 - dismean3) < 0.01:
                classOfPoint = 0
            if np.abs(dismean2 - dismean3) > 0 and np.abs(dismean2 - dismean3) < 0.01:
                classOfPoint = 0

        # ADD class of point
        pointClass[index] = classOfPoint


    # Creating a figure
    fig, axes = pl.subplots(1, 1)

    # plot points and their mean
    if wk != None:
        axes.plot([wix[0] for wix in wi], [wiy[1] for wiy in wi], 'ro',
                  [wjx[0] for wjx in wj], [wjy[1] for wjy in wj],'r*',
                  [wkx[0] for wkx in wk], [wky[1] for wky in wk], 'r^')

        # Plot means
        axes.plot(meanwi[0], meanwi[1], 'bo', label='mean o')
        axes.plot(meanwj[0], meanwj[1], 'b*', label='mean *')
        axes.plot(meanwk[0], meanwk[1], 'b^', label='mean ^')

    else:
        axes.plot([wix[0] for wix in wi], [wiy[1] for wiy in wi], 'ro',
                  [wjx[0] for wjx in wj], [wjy[1] for wjy in wj], 'r*')

        axes.plot(meanwi[0], meanwi[1], 'bo', label='mean o')
        axes.plot(meanwj[0], meanwj[1], 'b*', label='mean *')



    # convert array to 2D
    pointClass.shape = plot_resolution, plot_resolution

    # plot different areas which belongs to different classes
    axes.pcolormesh(X, Y, pointClass)

    axes.grid()
    pl.title(title + '\n Red points are samples, Blue points are Means\n'
                     +'Dotted lines are boundary among means'+
                      '\nArea with different colors belong to different classes')
    pl.legend()



# Dataset
dataset = {'w1':[[10,0],[0,-10],[5,-2]],
           'w2':[[5,10],[0,5],[5,5]],
           'w3':[[2,8],[-5,2],[10,-4]]}


# Part A
draw_decision_boundary_of_knn(wi=dataset['w1'], wj=dataset['w2']
                              , wk=None, title='W1 and W2')

# Part B
draw_decision_boundary_of_knn(wi=dataset['w1'], wj=dataset['w3']
                              , wk=None, title='W1 and W3')

# Part C
draw_decision_boundary_of_knn(wi=dataset['w2'], wj=dataset['w3']
                              , wk=None, title='W2 and W3')

# Part D
draw_decision_boundary_of_knn(wi=dataset['w1'], wj=dataset['w2']
                              , wk=dataset['w3'], title='W1, W2 and W3')

plt.show()