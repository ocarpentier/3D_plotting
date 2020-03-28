import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


x = np.linspace(0,10,50)
y = np.linspace(0,10,50)
x = np.sin(x)
z = np.linspace(0,10,50)
u = np.sin(x)
v = np.cos(x)
w = np.sin(x)
matrix = np.zeros((len(x),6))
for i in range(len(x)):
    matrix[i][0],matrix[i][1],matrix[i][2],matrix[i][3],matrix[i][4],matrix[i][5] = x[i],y[i],z[i],u[i],v[i],w[i]


class plotter:
    """
    :param  data:       a numpay array/matrix with six columns (order= x,y,z,u,v,w) first three are positionl arguments and last three are velocity arguments
    :param  title:      title (string)
    :param  xtitle:     title for the x-axis (string)
    :param  ytitle:     title for the y-axis (string)
    :arg    font:       change font of all three titles
    :arg:   colormp:    (see:https://matplotlib.org/3.1.1/tutorials/colors/colormaps.html)which colors to choose from
    :arg:   grid:       add grid or not to two dimensional plots
    :arg:   fontsize:   how big the titles must be
    :arg:   couleur:    If true there is color otherwise its uniform colored
    :arg:   ticks:      If False ticks get removed default is false
    :return:            a 2 dimensional plot in a self chosen plane of the airflow or with vectors or with streamlines or heatmap
    """

    #set a class variable this can be changed from outside the class
    planeinterval = 0.1

    def __init__(self,data,title,xtitle,ytitle,font=None, colormp=None,grid=None,fontsize=None,couleur=None,ticks=None):
        ## the required arguments
        self.data = data
        self.title = title
        self.xtitle = xtitle
        self.ytitle = ytitle

        ## optional arguments
        if font is None:
            self.font = None
        else:
            self.font = font

        if colormp is None:
            self.colormp = None
        else:
            self.colormp = colormp
        if grid is None:
            self.grid = False
        else:
            self.grid = True

        if fontsize is None:
            self.fontsize = 16
        else:
            self.fontsize = fontsize
        if couleur is None or couleur==False:
            self.couleur = False
        else:
            self.couleur = couleur
        if ticks is None:
            self.ticks = True
        else:
            self.ticks = ticks




    def vectorplaneyz(self, x):
        # select points for certain x +- the chosen interval
        data = self.data[(self.data[:, 0] < x + self.planeinterval) & (self.data[:, 0] > x - self.planeinterval)]

        # create 2D grid of the data point !!!!!!!!!!!!! should be left out for actual data
        X, Y = np.meshgrid(data[:, 1], data[:, 2])
        U, V = np.meshgrid(data[:, 4], data[:, 5])

        # X,Y = (data[:, 1], data[:, 2])
        # U,V = (data[:, 4], data[:, 5])

        #add different font
        csfont = {'fontname': self.font}

        ##plot data
        fig1, ax1 = plt.subplots()                                  ## make the figure
        ax1.set_title(self.title,fontsize=self.fontsize,**csfont)       ## add title
        plt.xlabel(self.xtitle,fontsize=self.fontsize,**csfont)     ## add title x-axis
        plt.ylabel(self.ytitle,fontsize=self.fontsize,**csfont)     ## add title y-axis
        plt.grid(self.grid)                                         ## if requested add grid

        # if requested add colour and yes its french colour values is based on speed
        if self.couleur:
            color = np.hypot(U, V)
            ax1.quiver(X, Y, U, V, color, cmap=self.colormp, alpha=1)   #actual plotting
        else:
            ax1.quiver(X, Y, U, V,alpha=1)                              #actual plotting

        #remove ticks on axes if requested
        if not self.ticks:
            ax1.xaxis.set_ticks([])         #removes bars on x-axis
            ax1.yaxis.set_ticks([])         # removes bars on y-axis
        plt.show()

    # ------------------------------------------------------------------------------
    def streamsplaneyz(self, x,density=None):

        # determine density of streamlines
        if density is None:
            density = 0.8
        else:
            density = density

        # select points for certain x
        data = self.data[(self.data[:, 0] < x + self.planeinterval) & (self.data[:, 0] > x - self.planeinterval)]

        # create 2D grid of the data point
        X, Y = np.meshgrid(data[:, 1], data[:, 2])
        U, V = np.meshgrid(data[:, 4], data[:, 5])

        # X,Y = (data[:, 1], data[:, 2])
        # U,V = (data[:, 4], data[:, 5])

        ## add font
        csfont = {'fontname': self.font}

        ##plot data
        fig1, ax1 = plt.subplots()
        # add color if requested
        if self.couleur:
            color = np.hypot(U, V)
            ax1.streamplot(X, Y, U, V,density=density, color=color,cmap=self.colormp)
        else:
            ax1.streamplot(X, Y, U, V,density=density)
        # add titles
        plt.title(self.title, fontsize=self.fontsize,**csfont)
        plt.xlabel(self.xtitle, fontsize=self.fontsize,**csfont)
        plt.ylabel(self.ytitle, fontsize=self.fontsize,**csfont)
        plt.grid(self.grid)

        # remove ticks on axes if requested
        if not self.ticks:
            ax1.xaxis.set_ticks([])  # removes bars on x-axis
            ax1.yaxis.set_ticks([])  # removes bars on y-axis

        plt.show()
        # -----------------------------------------
    def vectorplanexz(self, y):
        # select points for certain x +- the chosen interval
        data = self.data[(self.data[:, 1] < y + self.planeinterval) & (self.data[:, 1] > y - self.planeinterval)]

        # create 2D grid of the data point !!!!!!!!!!!!! should be left out for actual data
        X, Y = np.meshgrid(data[:, 0], data[:, 2])
        U, V = np.meshgrid(data[:, 3], data[:, 5])

        # X,Y = (data[:, 0], data[:, 2])
        # U,V = (data[:, 3], data[:, 5])

        # add different font
        csfont = {'fontname': self.font}

        ##plot data
        fig1, ax1 = plt.subplots()  ## make the figure
        ax1.set_title(self.title, fontsize=self.fontsize, **csfont)  ## add title
        plt.xlabel(self.xtitle, fontsize=self.fontsize, **csfont)  ## add title x-axis
        plt.ylabel(self.ytitle, fontsize=self.fontsize, **csfont)  ## add title y-axis
        plt.grid(self.grid)  ## if requested add grid

        # if requested add colour and yes its french colour values is based on speed
        if self.couleur:
            color = np.hypot(U, V)
            ax1.quiver(X, Y, U, V, color, cmap=self.colormp, alpha=1)  # actual plotting
        else:
            ax1.quiver(X, Y, U, V, alpha=1)  # actual plotting

        # remove ticks on axes if requested
        if not self.ticks:
            ax1.xaxis.set_ticks([])  # removes bars on x-axis
            ax1.yaxis.set_ticks([])  # removes bars on y-axis
        plt.show()
        # ------------------------------------------------------------------------------

    def streamsplanexz(self, y, density=None):
        # determine density of streamlines
        if density is None:
            density = 0.8
        else:
            density = density

        # select points for certain y
        data = self.data[(self.data[:, 1] < y + self.planeinterval) & (self.data[:, 1] > y - self.planeinterval)]

        # create 2D grid of the data point
        X, Y = np.meshgrid(data[:, 0], data[:, 2])
        U, V = np.meshgrid(data[:, 3], data[:, 5])

        # X,Y = (data[:, 0], data[:, 2])
        # U,V = (data[:, 3], data[:, 5])

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
        plt.grid(self.grid)
        # remove ticks on axes if requested
        if not self.ticks:
            ax1.xaxis.set_ticks([])  # removes bars on x-axis
            ax1.yaxis.set_ticks([])  # removes bars on y-axis

        plt.show()
        # -----------------------------------------

    def vectorplanexy(self, z):
        # select points for certain x +- the chosen interval
        data = self.data[(self.data[:, 2] < z + self.planeinterval) & (self.data[:, 2] > z - self.planeinterval)]

        # create 2D grid of the data point !!!!!!!!!!!!! should be left out for actual data
        X, Y = np.meshgrid(data[:, 0], data[:, 1])
        U, V = np.meshgrid(data[:, 3], data[:, 4])

        # X,Y = (data[:, 0], data[:, 1])
        # U,V = (data[:, 3], data[:, 4])

        # add different font
        csfont = {'fontname': self.font}

        ##plot data
        fig1, ax1 = plt.subplots()  ## make the figure
        ax1.set_title(self.title, fontsize=self.fontsize, **csfont)  ## add title
        plt.xlabel(self.xtitle, fontsize=self.fontsize, **csfont)  ## add title x-axis
        plt.ylabel(self.ytitle, fontsize=self.fontsize, **csfont)  ## add title y-axis
        plt.grid(self.grid)  ## if requested add grid

        # if requested add colour and yes its french colour values is based on speed
        if self.couleur:
            color = np.hypot(U, V)
            ax1.quiver(X, Y, U, V, color, cmap=self.colormp, alpha=1)  # actual plotting
        else:
            ax1.quiver(X, Y, U, V, alpha=1)  # actual plotting

        # remove ticks on axes if requested
        if not self.ticks:
            ax1.xaxis.set_ticks([])  # removes bars on x-axis
            ax1.yaxis.set_ticks([])  # removes bars on y-axis
        plt.show()

    #------------------------------------------------------------------------------
    def streamsplanexy(self, z, density=None):
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
        plt.grid(self.grid)
        # remove ticks on axes if requested
        if not self.ticks:
            ax1.xaxis.set_ticks([])  # removes bars on x-axis
            ax1.yaxis.set_ticks([])  # removes bars on y-axis

        plt.show()
        #-----------------------------------------
    def intersectingplanes(self, x, y):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        data1 = self.data[(self.data[:, 0] < x + self.planeinterval) & (self.data[:, 0] > x - self.planeinterval)]
        data2 = self.data[(self.data[:, 1] < y + self.planeinterval) & (self.data[:, 1] > y - self.planeinterval)]

        x1 = np.ones((1, len(data1[:, 3]))) * x
        X1, Y1 = np.meshgrid(x1, data1[:, 1])
        Z1, Y1 = np.meshgrid(data1[:, 2], data1[:, 1])
        v1, w1 = np.meshgrid(data1[:, 4], data1[:, 5])
        V1 = np.hypot(v1, w1)
        norm = mpl.colors.Normalize(vmin=0, vmax=1)

        x2 = list(data2[:, 0])
        z2 = list(data2[:, 2])
        u2 = list(data2[:, 3])
        w2 = list(data2[:, 5])
        deleting = True
        i = 0
        while deleting:
            if x2[i] > x:
                x2.pop(i)
                z2.pop(i)
                u2.pop(i)
                w2.pop(i)

            else:
                i += 1
            if i == len(x2):
                deleting = False
        y2 = np.ones((1, len(x2))) * y
        X2, Y2 = np.meshgrid(x2, y2)
        Z2, X2 = np.meshgrid(z2, x2)
        U2, W2 = np.meshgrid(u2, w2)
        V2 = np.hypot(U2, W2)

        x3 = list(data2[:, 0])
        z3 = list(data2[:, 2])
        u3 = list(data2[:, 3])
        w3 = list(data2[:, 5])
        deleting = True
        i = 0
        while deleting:
            if x3[i] <= x:
                x3.pop(i)
                z3.pop(i)
                u3.pop(i)
                w3.pop(i)

            else:
                i += 1
            if i == len(x3):
                deleting = False
        y3 = np.ones((1, len(x3))) * y
        X3, Y3 = np.meshgrid(x3, y3)
        Z3, X3 = np.meshgrid(z3, x3)
        U3, W3 = np.meshgrid(u3, w3)
        V3 = np.hypot(U3, W3)
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        ax.plot_surface(X1, Y1, Z1, facecolors=plt.cm.gist_rainbow(norm(V1)), shade=False)
        ax.plot_surface(X2, Y2, Z2, facecolors=plt.cm.gist_rainbow(norm(V2)), shade=False)
        ax.plot_surface(X3, Y3, Z3, facecolors=plt.cm.gist_rainbow(norm(V3)), shade=False)
        m = mpl.cm.ScalarMappable(cmap='gist_rainbow', norm=norm)
        m.set_array([])
        fig.colorbar(m)
        plt.show()

    def sphere_plane(self, x, colorsphere=None):
        if colorsphere is None:
            colorsphere = 'b'
        else:
            colorsphere = colorsphere
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        data1 = self.data[(self.data[:, 0] < x + self.planeinterval) & (self.data[:, 0] > x - self.planeinterval)]

        x1 = np.ones((1, len(data1[:, 3]))) * x
        X1, Y1 = np.meshgrid(x1, data1[:, 1])
        Z1, Y1 = np.meshgrid(data1[:, 2], data1[:, 1])
        v1, w1 = np.meshgrid(data1[:, 4], data1[:, 5])
        V1 = np.hypot(v1, w1)
        norm = mpl.colors.Normalize(vmin=0, vmax=1)

        r = 3  # radius [cm]
        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        x2 = np.cos(u) * np.sin(v)
        y2 = np.sin(u) * np.sin(v)
        z2 = np.cos(v)

        X2, Y2, Z2 = x2, y2, z2

        # X2,Y2 = np.meshgrid(x2,y2)
        # X2,Z2 = np.meshgrid(x2,z2)

        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        csfont = {'fontname': self.font}
        plt.title(self.title, fontsize=self.fontsize, **csfont)
        plt.xlabel(self.xtitle, fontsize=self.fontsize, **csfont)
        plt.ylabel(self.ytitle, fontsize=self.fontsize, **csfont)
        ax.plot_surface(X1, Y1, Z1, facecolors=plt.cm.gist_rainbow(norm(V1)), shade=False)
        ax.plot_surface(X2, Y2, Z2, color=colorsphere, shade=False)

        m = mpl.cm.ScalarMappable(cmap='gist_rainbow', norm=norm)
        m.set_array([])
        fig.colorbar(m)
        plt.show()

    ## !!!THOUGHT!!! ==> plotting the velocity field in three dimensions but only on the surface of tthe sphere


matrix = plotter(matrix,'Hello its meeeee','its a mee mario','jipse, zupt zee, pptse',colormp='gist_ncar',fontsize=11,font='Comic Sans MS',couleur=True)
matrix.planeinterval = 6
# matrix.streamsplaneyz(5,density=2)
# matrix.vectorplanexz(5)
# matrix.vectorplanexy(5)
# matrix.vectorplaneyz(5)
matrix.streamsplanexz(5,density=2)
matrix.streamsplanexy(5,density=2)