import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
from matplotlib import animation, rc
from IPython.display import HTML, Image
from matplotlib import rcParams





#creating data

np.random.seed(42)

N = 30
P = 1
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))

for fit_intercept in [True]:
    LR = LinearRegression(fit_intercept=fit_intercept)
    coef, mse1,all_coef = LR.fit_non_vectorised(X, y, 30, 100)

Y = y.values
x1 = X.values

if fit_intercept:
    x = np.ones((x1.shape[0], x1.shape[1] + 1))
    x[:, 1:] = x1
else:
    x = x1

rc('animation', html='html5')
rcParams['animation.convert_path'] = r'/usr/bin/convert'
fig, ax = plt.subplots()
ax.scatter(x1, Y)
line, = ax.plot([], [], lw=2)


def init():
    line.set_data([], [])
    return (line,)


def animate(i):
    x2 = x1
    y = np.dot(x, all_coef[i])
    line.set_data(x2, y)
    return line,


anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=10, interval=20, blit=True)
anim.save('animation_line_10.gif', writer='imagemagick', fps=30)

#plot contour

w0 = np.linspace(-(all_coef[0, 0] + 0.25), (all_coef[0, 0] + 0.25), 500)
w1 = np.linspace(-(all_coef[0, 1] + 0.25), (all_coef[0, 1] + 0.25), 500)
x = X.values
Y = y.values
x1 = np.ones((x.shape[0], x.shape[1] + 1))
x1[:, 1:] = x
W0, W1 = np.meshgrid(w0, w1)
mse = np.ones(W0.shape)
for i in range(W0.shape[0]):
    for j in range(W0.shape[1]):
        theta = np.array([W0[i, j], W1[i, j]])
        y_pred = np.dot(x1, theta)
        mse[i, j] = (np.sum((Y - y_pred) ** 2)) * 1 / (Y.shape[0])

rc('animation', html='html5')
fig_contour, ax = plt.subplots()
cp = ax.contour(W0, W1, mse)
ax.clabel(cp, inline=1, fontsize=10)
contour, = ax.plot([], [], lw=2)
def init_contour():
    contour.set_data([], [])
    return (contour,)
def animate_contour(i):
    x1 = all_coef[:i + 1, 0]
    x2 = all_coef[:i + 1, 1]
    contour.set_data(x1, x2)
    return (contour,)
anim_contour = animation.FuncAnimation(fig_contour, animate_contour, init_func=init_contour,frames=10, interval=20, blit=True)
#anim_contour
anim_contour.save('animation_contour_10.gif', writer='imagemagick', fps=30)



#3D Surface


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
title = ax.set_title('3D Test')
ax.plot_surface(W0, W1, mse, cmap='magma', edgecolor='none')
graph = ax.scatter([], [], [], c='r')


def update_graph(i):
    x1 = all_coef[1:i + 2, 0]
    x2 = all_coef[1:i + 2, 1]
    z = mse1[:i + 1]

    graph._offsets3d = (x1, x2, z)
    # title.set_text('3D Test, time={}'.format(num))


ani = animation.FuncAnimation(fig, update_graph, 10,interval=20, blit=False)
ani.save('animation_3Dsur_10.gif', writer='imagemagick', fps=30)


