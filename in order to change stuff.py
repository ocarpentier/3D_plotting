def streamsplanez(self, z, density=None):
    # determine density of streamlines
    if density is None:
        density = 0.8
    else:
        density = density

    # select points for certain y
    data = self.data[(self.data[:, 2] < z + self.planeinterval) & (self.data[:, 2] > z - self.planeinterval)]

    # create 2D grid of the data point
    X, Y = np.meshgrid(data[:, 0], data[:, 1])
    U, V = np.meshgrid(data[:, 3], data[:, 4])

    # X,Y = (data[:, 0], data[:, 1])
    # U,V = (data[:, 3], data[:, 4])

    ## add font
    csfont = {'fontname': self.font}

    ##plot data
    fig1, ax1 = plt.subplots()
    # add color if requested
    if self.couleur:
        color = np.hypot(U, V)
        ax1.streamplot(X, Y, U, V, density=density, color=color, cmap=self.colormp)
    else:
        ax1.streamplot(X, Y, U, V, density=density)
    # add titles
    plt.title(self.title, fontsize=self.fontsize, **csfont)
    plt.xlabel(self.xtitle, fontsize=self.fontsize, **csfont)
    plt.ylabel(self.ytitle, fontsize=self.fontsize, **csfont)

    # remove ticks on axes if requested
    if not self.ticks:
        ax1.xaxis.set_ticks([])  # removes bars on x-axis
        ax1.yaxis.set_ticks([])  # removes bars on y-axis

    plt.show()