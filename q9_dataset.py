import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
from metrics import *




np.random.seed(42)

N = 30
P = 5
X = np.random.randn(N, P)
y = pd.Series(np.random.randn(N))

x1=2*X[:,0]+X[:,1]
x1=x1.reshape(30,1)
X1=np.hstack((X,x1))

X=pd.DataFrame(X1)

for fit_intercept in [True,False]:
    LR = LinearRegression(fit_intercept=fit_intercept)
    theta,mse,theta_all=LR.fit_autograd(X, y,20) # here you can use fit_non_vectorised / fit_autograd methods
    y_hat = LR.predict(X)

    print('RMSE: ', rmse(y_hat, y))
    print('MAE: ', mae(y_hat, y))

