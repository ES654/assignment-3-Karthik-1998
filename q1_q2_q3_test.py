
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
from metrics import *

np.random.seed(42)

N = 30
P = 1
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))
batch_size=20

for fit_intercept in [False,True]:
    LR = LinearRegression(fit_intercept=fit_intercept)

    coef_, mse, all_coef = LR.fit_non_vectorised(X, y, batch_size)
    #coef_, mse, all_coef=LR.fit_vectorised(X, y, batch_size)
    #coef_, mse, all_coef = LR.fit_autograd(X, y, batch_size)# here you can use fit_non_vectorised / fit_autograd methods
    y_hat = LR.predict(X)

    print('RMSE: ', rmse(y_hat, y))
    print('MAE: ', mae(y_hat, y))




LR.plot_line_fit(X, y, coef_[0], coef_[1])
t_0 = all_coef[:, 0]
t_1 = all_coef[:, 1]
LR.plot_surface(X, y, t_0, t_1)
LR.plot_contour(X, y, t_0, t_1)

plt.show()