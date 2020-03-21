import numpy as np

def fWS(column):
    """
    :param column: column of the wingspan
    :return:weighed column
    """
    F = 0.1
    for i,xi in enumerate(column):
        if xi>60:
            column[i] = 0
        else:
            column[i] = xi*F
    return column





ac = ['','','','','','','','','','']
ac[0] = 'Airbus A330-200'
ac[1] = 'Airbus A330-300'
ac[2] = 'Boeing 737-800'
ac[3] = 'Boeing 747-400'
ac[4] = 'Boeing 777-200ER'
ac[5] = 'Boeing 777-300ER'
ac[6] = 'Boeing 787-9'
ac[7] = 'Boeing 787-10'
ac[8] = 'Embraer 175'
ac[9] = 'Embraer 190'


data = [ac,[[ 60.3, 58.37, 17.40, 880, 8800, 268, 233000, 7, 8, 8],[60.20, 63.69, 16.80, 880, 8200, 292, 233000, 8, 5, 8],
         [35.80, 39.47, 12.6, 850, 4200, 186, 73700, 3, 27, 6],[64.44, 70.67, 19.40, 920, 11500, 408, 390100, 9, 5, 6],[60.90, 63.80, 18.50, 900, 11800, 320, 297500, 5, 15, 10],
         [64.80, 73.86, 18.50,920, 12000, 408, 351543, 7, 14, 10],[60.10, 62.80, 16.30, 920, 11500, 294, 252650, 6, 13, 10],[60.10, 68.30, 17.02, 903, 12000, 344, 254100, 8, 5, 9],
         [26.00, 31.68, 9.86, 850, 3300, 88, 36500, 2, 17, 4],[28.72, 36.24, 10.55, 850, 3300, 100, 45000, 2, 32, 4]]]

matrix = np.zeros((10,10))
for i in range(len(data[1])):
    for j in range(len(data[1])):
        matrix[i][j] = data[1][i][j]

print(fWS(matrix[:,0]))
matrix[:,0] = fWS(matrix[:,0])
print(matrix)






