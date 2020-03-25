import numpy as np
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
            self.ticks = False




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

        # remove ticks on axes if requested
        if not self.ticks:
            ax1.xaxis.set_ticks([])  # removes bars on x-axis
            ax1.yaxis.set_ticks([])  # removes bars on y-axis

        plt.show()
        #-----------------------------------------


        ## !!!THOUGHT!!! ==> plotting the velocity field in three dimensions but only on the surface of tthe sphere


matrix = plotter(matrix,'Hello its meeeee','its a mee mario','jipse, zupt zee, pptse',colormp='gist_ncar',grid=True,fontsize=11,font='Ariel Black',couleur=True)
matrix.planeinterval = 6
matrix.streamsplaneyz(5,density=2)
# matrix.vectorplanexz(5)
# matrix.vectorplanexy(5)
# matrix.vectorplaneyz(5)
# matrix.vectorplaneyxz(5)
# matrix.vectorplanexy(5)