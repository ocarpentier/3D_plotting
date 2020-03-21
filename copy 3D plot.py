import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
# from mayavi.mlab import *
#
# class plotter:
#     def __init__(self,data):
#         self.data = data

x = np.linspace(0,10,10)
y = np.linspace(10,0,10)
z = np.linspace(0,10,10)
u = np.sin(x)
v = np.cos(x)
w = np.tan(x)
matrix = np.zeros((len(x),6))
for i in range(len(x)):
    matrix[i][0],matrix[i][1],matrix[i][2],matrix[i][3],matrix[i][4],matrix[i][5] = x[i],y[i],z[i],u[i],v[i],w[i]

class plotter:
    """

    """
    interval = 0.01
    def __init__(self):
        self


    def plotplane(self,z):
        # put the array in dataframe
        df = pd.DataFrame(self)


        #select points for certain z
        # df = df[(df[2] > z-self.interval) & (df[2]< z+self.interval)]
        print(df[:,0])

        ##plot data
        fig1, ax1 = plt.subplots()
        ax1.set_title('Arrows scale with plot width, not view')
        Q = ax1.quiver(df[:,0], df[:,1], df[:,3], df[:,4], units='width')
        qk = ax1.quiverkey(Q, 0.9, 0.9, 2, r'$2 \frac{m}{s}$', labelpos='E',coordinates='figure')
        plt.show()

matrix.plotter
matrix.plotplane(0)




###---------------------------------------------------------------------
# copy erda 4c

