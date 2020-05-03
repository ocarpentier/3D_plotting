import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rc


x = np.linspace(0,10,50)
y = np.linspace(0,10,50)
z = np.linspace(0,10,50)
u = np.sin(x)
v = np.sin(x)
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
    def intersectingplanesxy(self, x, y,totalv=None):
        if totalv is None:
            totalv = True
        else:
            totalv = totalv
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        data1 = self.data[(self.data[:, 0] < x + self.planeinterval) & (self.data[:, 0] > x - self.planeinterval)]
        data2 = self.data[(self.data[:, 1] < y + self.planeinterval) & (self.data[:, 1] > y - self.planeinterval)]

        x1 = np.ones((1, len(data1[:, 3]))) * x
        X1, Y1 = np.meshgrid(x1, data1[:, 1])
        Z1, Y1 = np.meshgrid(data1[:, 2], data1[:, 1])
        v1, w1 = np.meshgrid(data1[:, 4], data1[:, 5])
        u1,v1  = np.meshgrid(data1[:, 3], data1[:, 4])
        if totalv :
            Ve1 = np.hypot(np.hypot(v1, w1),u1)
        else:
            Ve1 = np.hypot(v1, w1)


        x2 = list(data2[:, 0])
        z2 = list(data2[:, 2])
        u2 = list(data2[:, 3])
        v2 = list(data2[:, 4])
        w2 = list(data2[:, 5])
        deleting = True
        i = 0
        passing = False
        while deleting:
            if x2[i] > x:
                x2.pop(i)
                z2.pop(i)
                u2.pop(i)
                v2.pop(i)
                w2.pop(i)
                passing = True
            else:
                i += 1
            if i == len(x2):
                deleting = False
        y2 = np.ones((1, len(x2))) * y
        X2, Y2 = np.meshgrid(x2, y2)
        Z2, X2 = np.meshgrid(z2, x2)
        U2, W2 = np.meshgrid(u2, w2)
        U2, V2 = np.meshgrid(u2, v2)

        if totalv:
            Ve2 = np.hypot(np.hypot(U2, W2),V2)
        else:
            Ve2 = np.hypot(U2, W2)

        x3 = list(data2[:, 0])
        z3 = list(data2[:, 2])
        u3 = list(data2[:, 3])
        v3 = list(data2[:, 4])
        w3 = list(data2[:, 5])
        deleting = True
        i = 0
        while deleting and passing:
            if x3[i] <= x:
                x3.pop(i)
                z3.pop(i)
                u3.pop(i)
                v3.pop(i)
                w3.pop(i)

            else:
                i += 1
            if i == len(x3):
                deleting = False
        if passing:
            y3 = np.ones((1, len(x3))) * y
            X3, Y3 = np.meshgrid(x3, y3)
            Z3, X3 = np.meshgrid(z3, x3)
            U3, W3 = np.meshgrid(u3, w3)
            U3, V3 = np.meshgrid(u3, v3)
            if totalv:
                Ve3 = np.hypot(np.hypot(U3, W3),V3)
            else:
                Ve3 = np.hypot(U3, W3)
            a = max(np.amax(Ve3), np.amax(Ve2), np.amax(Ve1))
            b = min(np.amin(Ve3), np.amin(Ve2), np.amin(Ve1))
        else:
            a = max( np.amax(Ve2), np.amax(Ve1))
            b = min(np.amin(Ve2), np.amin(Ve1))
        norm = mpl.colors.Normalize(vmin=b, vmax=a)
        ax.plot_surface(X1, Y1, Z1, facecolors=plt.cm.prism(norm(Ve1)), shade=False)
        ax.plot_surface(X2, Y2, Z2, facecolors=plt.cm.prism(norm(Ve2)), shade=False)
        if passing:
         ax.plot_surface(X3, Y3, Z3, facecolors=plt.cm.prism(norm(Ve3)), shade=False)
        m = mpl.cm.ScalarMappable(cmap='gist_rainbow', norm=norm)
        m.set_array([])
        fig.colorbar(m)
        plt.show()

    def intersectingplanesyz(self, z, y, totalv=None):
        if totalv is None:
            totalv = True
        else:
            totalv = totalv
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        data1 = self.data[(self.data[:, 0] < z + self.planeinterval) & (self.data[:, 0] > z - self.planeinterval)]
        data2 = self.data[(self.data[:, 1] < y + self.planeinterval) & (self.data[:, 1] > y - self.planeinterval)]

        z1 = np.ones((1, len(data1[:, 3]))) * z
        Z1, Y1 = np.meshgrid(z1, data1[:, 1])
        X1, Y1 = np.meshgrid(data1[:, 0], data1[:, 1])
        v1, w1 = np.meshgrid(data1[:, 4], data1[:, 5])
        u1, v1 = np.meshgrid(data1[:, 3], data1[:, 4])
        if totalv:
            Ve1 = np.hypot(np.hypot(v1, w1), u1)
        else:
            Ve1 = np.hypot(v1, u1)

        x2 = list(data2[:, 0])
        z2 = list(data2[:, 2])
        u2 = list(data2[:, 3])
        v2 = list(data2[:, 4])
        w2 = list(data2[:, 5])
        deleting = True
        i = 0
        passing = False
        while deleting:
            if x2[i] > x:
                x2.pop(i)
                z2.pop(i)
                u2.pop(i)
                v2.pop(i)
                w2.pop(i)
                passing = True
            else:
                i += 1
            if i == len(x2):
                deleting = False
        y2 = np.ones((1, len(x2))) * y
        X2, Y2 = np.meshgrid(x2, y2)
        Z2, X2 = np.meshgrid(z2, x2)
        U2, W2 = np.meshgrid(u2, w2)
        U2, V2 = np.meshgrid(u2, v2)

        if totalv:
            Ve2 = np.hypot(np.hypot(U2, W2), V2)
        else:
            Ve2 = np.hypot(U2, W2)

        x3 = list(data2[:, 0])
        z3 = list(data2[:, 2])
        u3 = list(data2[:, 3])
        v3 = list(data2[:, 4])
        w3 = list(data2[:, 5])
        deleting = True
        i = 0
        while deleting and passing:
            if x3[i] <= x:
                x3.pop(i)
                z3.pop(i)
                u3.pop(i)
                v3.pop(i)
                w3.pop(i)

            else:
                i += 1
            if i == len(x3):
                deleting = False
        if passing:
            y3 = np.ones((1, len(x3))) * y
            X3, Y3 = np.meshgrid(x3, y3)
            Z3, X3 = np.meshgrid(z3, x3)
            U3, W3 = np.meshgrid(u3, w3)
            U3, V3 = np.meshgrid(u3, v3)
            if totalv:
                Ve3 = np.hypot(np.hypot(U3, W3), V3)
            else:
                Ve3 = np.hypot(U3, W3)
            a = max(np.amax(Ve3), np.amax(Ve2), np.amax(Ve1))
            b = min(np.amin(Ve3), np.amin(Ve2), np.amin(Ve1))
        else:
            a = max(np.amax(Ve2), np.amax(Ve1))
            b = min(np.amin(Ve2), np.amin(Ve1))
        norm = mpl.colors.Normalize(vmin=b, vmax=a)
        ax.plot_surface(X1, Y1, Z1, facecolors=plt.cm.prism(norm(Ve1)), shade=False)
        ax.plot_surface(X2, Y2, Z2, facecolors=plt.cm.prism(norm(Ve2)), shade=False)
        if passing:
            ax.plot_surface(X3, Y3, Z3, facecolors=plt.cm.prism(norm(Ve3)), shade=False)
        m = mpl.cm.ScalarMappable(cmap='gist_rainbow', norm=norm)
        m.set_array([])
        fig.colorbar(m)
        plt.show()


    def intersectingplanesxz(self, x, z,totalv=None):
        if totalv is None:
            totalv = True
        else:
            totalv = totalv
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        data1 = self.data[(self.data[:, 0] < x + self.planeinterval) & (self.data[:, 0] > x - self.planeinterval)]
        data2 = self.data[(self.data[:, 2] < z + self.planeinterval) & (self.data[:, 2] > z - self.planeinterval)]

        x1 = np.ones((1, len(data1[:, 3]))) * x
        X1, Y1 = np.meshgrid(x1, data1[:, 1])
        Z1, Y1 = np.meshgrid(data1[:, 2], data1[:, 1])
        v1, w1 = np.meshgrid(data1[:, 4], data1[:, 5])
        u1,v1  = np.meshgrid(data1[:, 3], data1[:, 4])
        if totalv :
            Ve1 = np.hypot(np.hypot(v1, w1),u1)
        else:
            Ve1 = np.hypot(v1, w1)


        x2 = list(data2[:, 0])
        y2 = list(data2[:, 1])
        u2 = list(data2[:, 3])
        v2 = list(data2[:, 4])
        w2 = list(data2[:, 5])
        deleting = True
        i = 0
        passing = False
        while deleting:
            if x2[i] > x:
                x2.pop(i)
                z2.pop(i)
                u2.pop(i)
                v2.pop(i)
                w2.pop(i)
                passing = True
            else:
                i += 1
            if i == len(x2):
                deleting = False
        z2 = np.ones((1, len(x2))) * z
        X2, Y2 = np.meshgrid(x2, y2)
        Z2, X2 = np.meshgrid(z2, x2)
        U2, W2 = np.meshgrid(u2, w2)
        U2, V2 = np.meshgrid(u2, v2)

        if totalv:
            Ve2 = np.hypot(np.hypot(U2, W2),V2)
        else:
            Ve2 = np.hypot(U2, V2)

        x3 = list(data2[:, 0])
        y3 = list(data2[:, 1])
        u3 = list(data2[:, 3])
        v3 = list(data2[:, 4])
        w3 = list(data2[:, 5])
        deleting = True
        i = 0
        while deleting and passing:
            if x3[i] <= x:
                x3.pop(i)
                y3.pop(i)
                u3.pop(i)
                v3.pop(i)
                w3.pop(i)

            else:
                i += 1
            if i == len(x3):
                deleting = False
        if passing:
            z3 = np.ones((1, len(x3))) * z
            X3, Y3 = np.meshgrid(x3, y3)
            Z3, X3 = np.meshgrid(z3, x3)
            U3, W3 = np.meshgrid(u3, w3)
            U3, V3 = np.meshgrid(u3, v3)
            if totalv:
                Ve3 = np.hypot(np.hypot(U3, W3),V3)
            else:
                Ve3 = np.hypot(U3, V3)
            a = max(np.amax(Ve3), np.amax(Ve2), np.amax(Ve1))
            b = min(np.amin(Ve3), np.amin(Ve2), np.amin(Ve1))
        else:
            a = max( np.amax(Ve2), np.amax(Ve1))
            b = min(np.amin(Ve2), np.amin(Ve1))
        norm = mpl.colors.Normalize(vmin=b, vmax=a)
        ax.plot_surface(X1, Y1, Z1, facecolors=plt.cm.prism(norm(Ve1)), shade=False)
        ax.plot_surface(X2, Y2, Z2, facecolors=plt.cm.prism(norm(Ve2)), shade=False)
        if passing:
         ax.plot_surface(X3, Y3, Z3, facecolors=plt.cm.prism(norm(Ve3)), shade=False)
        m = mpl.cm.ScalarMappable(cmap='gist_rainbow', norm=norm)
        m.set_array([])
        fig.colorbar(m)
        plt.show()

    def sphere_plane(self, x,xs,ys,zs,r, colorsphere=None):
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


        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        x2 = r*np.cos(u) * np.sin(v)
        y2 = r*np.sin(u) * np.sin(v)
        z2 = r*np.cos(v)
        x2 += xs
        y2 += ys
        z2 += zs

        X2, Y2, Z2 = x2, y2, z2


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
#
# fig1 = plotter(matrix,'Test Title','x-axis','y-axis',colormp='gist_ncar',fontsize=11,font='Comic Sans MS',couleur=True,grid=None,ticks=None)
# fig2 = plotter(matrix,'Test Title','x-axis','y-axis',colormp='hsv',fontsize=15,font=None,couleur=False,grid=True,ticks=True)
# fig1.planeinterval = 10.1                 # Most likely there wont be many points having exactly the same value hence all the points lying in the interval will be projected on the plane this is how its adjusted
# fig2.planeinterval = 10.1                 # same
# fig1.intersectingplanes(3,5)

"""
first set a new variable name equal to the class with as first argument the dataset. This should be a 
numpy array with six columns with no strings in it(first entry= x-location, second y-loc,third z,
 fourth u,fifth v and sixth w). Then you call  "variable" + "." + "plotting function".
 
 I have produced some random test data for you guys this needed a meshing 
 function perhaps for the real data this is not necessary and could create some strange results in this 
 case  just tell me its a quick fix. i'll give some examples of the function and 
 brief explanation below you could import this file like you do with numpy or copy paste it but do not
  write in this file so everyone can keep using it easily.  
  
  the dataset and 3 titles are obligatory to fill in the rest are optional arguments
  
  if you'd like other plotting methods or change certain things you can always ask me(= Oscar).
"""

# fig1 = plotter(matrix,'Test Title','x-axis','y-axis',colormp='gist_ncar',fontsize=11,font='Comic Sans MS',couleur=True,grid=None,ticks=None)
# fig2 = plotter(matrix,'Test Title','x-axis','y-axis',colormp='hsv',fontsize=15,font=None,couleur=False,grid=True,ticks=True)
# fig1.planeinterval = 10.1                 # Most likely there wont be many points having exactly the same value hence all the points lying in the interval will be projected on the plane this is how its adjusted
# fig2.planeinterval = 10.1                 # same

# fig1.vectorplanexz(5)                   # does not look good for a lot of datapoints
# fig1.streamsplanexz(5,density=3)        # density is an optional argument
# fig1.streamsplanexy(5,density=0.5)      # you can choose the plane your in
# fig1.intersectingplanesxy(5,5,totalv=None)  # if planevelocity is True then only the colormap will be based on the in plane velocities. Here i didn't find a way to adapt the color scheme of the heat map with a variable for this you need to enter the class and change it manually or aske me :). this plot looks wrong but i think ith the acquired data it wont be a problem
# fig1.sphere_plane(x=5,xs=2,ys=3,zs=4,r=2,colorsphere='c')    # you can choose the color of your sphere this is also optional. It doesnt look like a sphere but  ithink it is due to the proportions of the axes and it'll probably resolve itself
# #
# fig2.vectorplanexz(5)
# fig2.streamsplanexz(5)
# fig2.streamsplanexy(5,density=0.5)
# fig2.intersectingplanes(0,5)
# fig2.sphere_plane(x=5,xs=2,ys=3,zs=-1,r=2,colorsphere='black')

## !!!Draw it in different order (intersecting planes)!!!

