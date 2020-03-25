import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 10)
y = np.linspace(0, 10, 10)
x = np.sin(x)
z = np.linspace(0, 10, 10)
u = np.sin(x)
v = np.cos(x)
w = np.sin(x)
matrix = np.zeros((len(x), 6))
for i in range(len(x)):
    matrix[i][0], matrix[i][1], matrix[i][2], matrix[i][3], matrix[i][4], matrix[i][5] = x[i], y[i], z[i], u[i], v[i], \
                                                                                         w[i]
print(matrix)


class plotter:
    """
    input:  a numpay array/matrix
    output: a 2 dimensional plot in a self chosen plane of the airflow
    """
    planeinterval = 0.1

    def __init__(self, font=None, colormp=None):

    def vectorplanex(self, x):
        # choose an interval which is projected on chosen y coordinata
        interval = 10.01
        data = self

        # select points for certain z
        data = data[(data[:, 2] < x + interval) & (data[:, 2] > x - interval)]

        # create 2D grid of the data point
        X, Y = np.meshgrid(data[:, 1], data[:, 2])
        U, V = np.meshgrid(data[:, 4], data[:, 5])

        # colour based on the total speed
        color = np.hypot(U, V)

        ##plot data
        fig1, ax1 = plt.subplots()
        ax1.set_title('Arrows scale with plot width, not view')
        ax1.quiver(X, Y, U, V, color, alpha=0.8)
        ax1.set_aspect('equal')  # sets aspect ratio to 1:1
        # ax1.xaxis.set_ticks([])         #removes bars on axis
        # ax1.yaxis.set_ticks([])         # removes bars on axis

        plt.show()

    # ------------------------------------------------------------------------------
    def streamsplanex(self, x):
        # choose an interval which is projected on chosen y coordinata
        interval = 10.01
        data = self

        # select points for certain z
        data = data[(data[:, 2] < x + interval) & (data[:, 2] > x - interval)]

        # create 2D grid of the data point
        X, Y = np.meshgrid(data[:, 0], data[:, 2])
        U, V = np.meshgrid(data[:, 3], data[:, 5])

        # colour based on the total speed
        color = np.hypot(U, V)

        ##plot data
        fig1, ax1 = plt.subplots()
        ax1.set_title('Arrows scale with plot width, not view')
        ax1.streamplot(X, Y, U, V, density=0.8)
        ax1.set_aspect('equal')  # sets aspect ratio to 1:1
        # ax1.xaxis.set_ticks([])         #removes bars on axis
        # ax1.yaxis.set_ticks([])         # removes bars on axis

        plt.show()
        # -----------------------------------------

    def vectorplaney(self, y):
        # choose an interval which is projected on chosen y coordinata
        interval = 10.01
        data = self

        # select points for certain z
        data = data[(data[:, 2] < y + interval) & (data[:, 2] > y - interval)]

        # create 2D grid of the data point
        X, Y = np.meshgrid(data[:, 0], data[:, 2])
        U, V = np.meshgrid(data[:, 3], data[:, 5])

        # colour based on the total speed
        color = np.hypot(U, V)

        ##plot data
        fig1, ax1 = plt.subplots()
        ax1.set_title('Arrows scale with plot width, not view')
        ax1.quiver(X, Y, U, V, color, alpha=0.8)
        ax1.set_aspect('equal')  # sets aspect ratio to 1:1
        # ax1.xaxis.set_ticks([])         #removes bars on axis
        # ax1.yaxis.set_ticks([])         # removes bars on axis

        plt.show()

        # ------------------------------------------------------------------------------

    def streamsplaney(self, y):
        # choose an interval which is projected on chosen y coordinata
        interval = 10.01
        data = self

        # select points for certain z
        data = data[(data[:, 2] < y + interval) & (data[:, 2] > y - interval)]

        # create 2D grid of the data point
        X, Y = np.meshgrid(data[:, 0], data[:, 2])
        U, V = np.meshgrid(data[:, 3], data[:, 5])

        # colour based on the total speed
        color = np.hypot(U, V)

        ##plot data
        fig1, ax1 = plt.subplots()
        ax1.set_title('Arrows scale with plot width, not view')
        ax1.streamplot(X, Y, U, V, density=0.8)
        ax1.set_aspect('equal')  # sets aspect ratio to 1:1
        # ax1.xaxis.set_ticks([])         #removes bars on axis
        # ax1.yaxis.set_ticks([])         # removes bars on axis
        # qk = ax1.quiverkey(Q, 0.9, 0.9, 2, r'$2 \frac{m}{s}$', labelpos='E',coordinates='figure')
        plt.show()
        # -----------------------------------------

    def vectorplanez(self, z):
        # choose an interval which is projected on chosen y coordinata
        interval = 10.01
        data = self

        # select points for certain z
        data = data[(data[:, 2] < z + interval) & (data[:, 2] > z - interval)]

        # create 2D grid of the data point
        X, Y = np.meshgrid(data[:, 0], data[:, 1])
        U, V = np.meshgrid(data[:, 3], data[:, 4])

        # colour based on the total speed
        color = np.hypot(U, V)

        ##plot data
        fig1, ax1 = plt.subplots()
        ax1.set_title('Arrows scale with plot width, not view')
        ax1.quiver(X, Y, U, V, color, alpha=0.8)
        ax1.set_aspect('equal')  # sets aspect ratio to 1:1
        # ax1.xaxis.set_ticks([])         #removes bars on axis
        # ax1.yaxis.set_ticks([])         # removes bars on axis

        plt.show()

    # ------------------------------------------------------------------------------
    def streamsplanez(self, z):
        # choose an interval which is projected on chosen y coordinata
        interval = 10.01
        data = self

        # select points for certain z
        data = data[(data[:, 2] < z + interval) & (data[:, 2] > z - interval)]

        # create 2D grid of the data point
        X, Y = np.meshgrid(data[:, 0], data[:, 1])
        U, V = np.meshgrid(data[:, 3], data[:, 4])

        # colour based on the total speed
        color = np.hypot(U, V)

        ##plot data
        fig1, ax1 = plt.subplots()
        ax1.set_title('Arrows scale with plot width, not view')
        ax1.streamplot(X, Y, U, V, density=0.8)
        ax1.set_aspect('equal')  # sets aspect ratio to 1:1
        # ax1.xaxis.set_ticks([])         #removes bars on axis
        # ax1.yaxis.set_ticks([])         # removes bars on axis

        plt.show()
        # -----------------------------------------

        ## !!!THOUGHT!!! ==> plotting the velocity field in three dimensions but only on the surface of tthe sphere


plotter.streamsplaney(matrix, 1.111)
