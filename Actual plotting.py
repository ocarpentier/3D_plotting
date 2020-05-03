import ThreeDplot
import numpy as np
import glob as gl
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rc
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff

def gen_streamline(u,v,w,x1,y1,z1,forward=None):
    if forward is None:
        forward = True
    else:
        forward = forward
    if forward:
        matrix1 = np.array([0,w,-v],
                          [-w,0,u],
                          [v,-u,0])
        matrix2 = np.array([-w*y1+v*z1,
                            -u*z1+w*x1,
                            -v*x1+u*y1])
        matrix3 = np.linalg.solve(matrix1,matrix2)

    return matrix3



df = pd.read_excel(f'zzero.xlsx')
# df = df[df.index % 5 == 0]  # Selects every 3rd raw starting from 0
df = np.array(df)
print(df.shape)
df = df[(df[:,1]<1) & (df[:,1] > -1)]
df = df[1:-1]
print(df.shape)
print(np.amax(df))
# fig1 = ThreeDplot.plotter(df,'Title','x-title','y-title',couleur=False,colormp='gist_ncar')
# fig1.planeinterval = 0.1
# fig1.streamsplanexy(0,density=1)
#















# dataarray1 = np.zeros((479,6))
# # dataarray2 = np.zeros((int((13594+46195+3)/5)+1,6))
#
# # files = gl.glob(f'x=-10_projected/-10_*.csv')
# count = -1
# # for file in files:
# #     f = open(file,'r')
# #     lines = f.readlines()
# #     lines = lines[1:-1]
# #     f.close()
# #     for line in lines:
# #         count += 1
# #         line = line[0:-1]
# #         line = line.split(',')
# #
# #
# #         dataarray1[count][0] = float(line[3])
# #         dataarray1[count][1] = float(line[4])
# #         dataarray1[count][2] = float(line[5])
# #         dataarray1[count][3] = float(line[6])
# #         dataarray1[count][4] = float(line[7])
# #         dataarray1[count][5] = float(line[8])
#
# f = open('y=0_projected/8_0_7.csv','r')
# lines = f.readlines()
# lines = lines[1:-1]
# f.close()
# for line in lines:
#     count += 1
#     line = line[0:-1]
#     line = line.split(',')
#
#
#     dataarray1[count][0] = float(line[3])
#     dataarray1[count][1] = float(line[4])
#     dataarray1[count][2] = float(line[5])
#     dataarray1[count][3] = float(line[6])
#     dataarray1[count][4] = float(line[7])
#     dataarray1[count][5] = float(line[8])
# # print(count)
# # files = gl.glob(f'y=0_projected/*.csv')
# #
# # for file in files:
# #     f = open(file,'r')
# #     lines = f.readlines()
# #     lines = lines[1:-1]
# #     f.close()
# #     for line in lines:
# #         count += 1
# #         line = line[0:-1]
# #         line = line.split(',')
# #
# #
# #         dataarray1[count][0] = float(line[3])
# #         dataarray1[count][1] = float(line[4])
# #         dataarray1[count][2] = float(line[5])
# #         dataarray1[count][3] = float(line[6])
# #         dataarray1[count][4] = float(line[7])
# #         dataarray1[count][5] = float(line[8])
# #
# # for i  in range(len(dataarray1)):
# #     print(i)
# #     if i%5==0:
# #         dataarray2[int(i/5)] = dataarray1[i]
# # fig1 = ThreeDplot.plotter(dataarray1,'Title', 'x','y',colormp='gist_ncar',couleur=True)
# # fig1.intersectingplanes(,0,totalv=True)
# # fig1 = ThreeDplot.plotter(dataarray1,'Title', 'x','y',colormp='gist_ncar',couleur=True)
# # fig1.planeinterval = 10
# # fig1.vectorplaneyz(-10)
#
# data = dataarray1
# couleur = True
# density = 1.5
# colormp = 'gist_ncar'
# X,Y = np.meshgrid(data[:, 0], data[:, 2])
# U,V = np.meshgrid(data[:, 3], data[:, 5])
# fig1, ax1 = plt.subplots()
# # add color if requested
# if couleur:
#     color = np.hypot(U, V)
#     ax1.streamplot(X, Y, U, V,density=density, color=color,cmap=colormp)
# else:
#     ax1.streamplot(X, Y, U, V,density=density)
# # add titles
# plt.title('Airflow in the yz plane wit x= -10')
# plt.xlabel('y-position [mm]')
# plt.ylabel('z-position [mm]')
#
#
#
# plt.show()