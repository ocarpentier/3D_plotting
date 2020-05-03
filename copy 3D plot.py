def intersectingplanes(self, x, y, planevelocity=None):
    if planevelocity is None:
        planevelocity = False

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    data1 = self.data[(self.data[:, 0] < x + self.planeinterval) & (self.data[:, 0] > x - self.planeinterval)]

    data2 = self.data[(self.data[:, 1] < y + self.planeinterval) & (self.data[:, 1] > y - self.planeinterval)]

    x1 = np.ones((1, len(data1[:, 3]))) * x
    X1, Y1 = np.meshgrid(x1, data1[:, 1])
    Z1, Y1 = np.meshgrid(data1[:, 2], data1[:, 1])
    v1, w1 = np.meshgrid(data1[:, 4], data1[:, 5])
    u1, v1 = np.meshgrid(data1[:, 3], data1[:, 5])
    if not planevelocity:
        Ve1 = np.hypot(np.hypot(v1, w1), u1)
    else:
        Ve1 = np.hypot(v1, w1)

    x2 = list(data2[:, 0])
    y2 = np.ones((1, len(x2))) * y
    z2 = list(data2[:, 2])
    u2 = list(data2[:, 3])
    v2 = list(data2[:, 4])
    w2 = list(data2[:, 5])
    X, Y = list(np.meshgrid(x2, y2))
    Z, X = list(np.meshgrid(z2, x2))
    U, W = list(np.meshgrid(u2, w2))
    U, V = list(np.meshgrid(u2, v2))
    for i in range(len(X)):
        X[i] = list(X[i])
        Y[i] = list(Y[i])
        Z[i] = list(Z[i])
        U[i] = list(U[i])
        V[i] = list(V[i])
        W[i] = list(W[i])
    X2, Y2, Z2, U2, V2, W2 = X, Y, Z, U, V, W
    deleting = True
    i = 0
    j = 0
    while deleting:
        if X2[i][j] > x:
            X2[i].pop(j)
            Y2[i].pop(j)
            Z2[i].pop(j)
            U2[i].pop(j)
            V2[i].pop(j)
            W2[i].pop(j)
            passing = True
        elif j == len(X2[i]) - 1:
            i += 1
            j = 0
        elif j == len(X2[i]) - 1 and i == len(X2) - 1:
            deleting = False

        else:
            j += 1
    X2, Y2, Z2, U2, V2, W2 = np.array(X2), np.array(Y2), np.array(Z2), np.array(U2), np.array(V2), np.array(W2)
    if not planevelocity:
        Ve2 = np.hypot(np.hypot(U2, W2), V2)
    else:
        Ve2 = np.hypot(U2, W2)

    X3, Y3, Z3, U3, V3, W3 = X, Y, Z, U, V, W
    deleting = True
    i = 0
    j = 0
    while deleting:
        if X3[i][j] > x:
            X3[i].pop(j)
            Y3[i].pop(j)
            Z3[i].pop(j)
            U3[i].pop(j)
            V3[i].pop(j)
            W3[i].pop(j)
            passing = True
        elif j == len(X3[i]):
            i += 1
            j = 0
        elif j == len(X3[i]) and i == len(X3):
            deleting = False

        else:
            j += 1
    X3, Y3, Z3, U3, V3, W3 = np.array(X3), np.array(Y3), np.array(Z3), np.array(U3), np.array(V3), np.array(W3)
    if not planevelocity:
        Ve3 = np.hypot(np.hypot(U3, W3), V3)
    else:
        Ve3 = np.hypot(U3, W3)
        a = max(np.amax(Ve3), np.amax(Ve2), np.amax(Ve1))
        b = min(np.amin(Ve3), np.amin(Ve2), np.amin(Ve1))

    norm = mpl.colors.Normalize(vmin=b, vmax=a)
    ax.plot_surface(X1, Y1, Z1, facecolors=plt.cm.gist_rainbow(norm(Ve1)), shade=False)
    ax.plot_surface(X2, Y2, Z2, facecolors=plt.cm.gist_rainbow(norm(Ve2)), shade=False)
    if passing:
        ax.plot_surface(X3, Y3, Z3, facecolors=plt.cm.gist_rainbow(norm(Ve3)), shade=False)
    m = mpl.cm.ScalarMappable(cmap='gist_rainbow', norm=norm)
    m.set_array([])
    fig.colorbar(m)
    plt.show()