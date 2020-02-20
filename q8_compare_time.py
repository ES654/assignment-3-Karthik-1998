import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
import time



np.random.seed(42)

time_normal_P=[]
time_grad_P=[]

for P in range(1,6):
    N = 30
    X = pd.DataFrame(np.random.randn(N, P))
    y = pd.Series(np.random.randn(N))

    fit_intercept = True
    LR = LinearRegression(fit_intercept=fit_intercept)
    start_seconds = time.time()
    coef_grad,mse,all_coef=LR.fit_autograd(X, y,20) # here you can use fit_non_vectorised / fit_autograd methods
    end_seconds =time.time()

    start_seconds_normal = time.time()
    coef_normal=LR.fit_normal(X, y) # here you can use fit_non_vectorised / fit_autograd methods
    end_seconds_normal=time.time()
    time_normal_P.append(end_seconds_normal-start_seconds_normal)
    time_grad_P.append(end_seconds-start_seconds)

time_normal_N=[]
time_grad_N=[]
for N in range(20,70,10):
    #N = 30
    X = pd.DataFrame(np.random.randn(N, P))
    y = pd.Series(np.random.randn(N))
    P=5


    fit_intercept = True
    LR = LinearRegression(fit_intercept=fit_intercept)
    start_seconds_N = time.time()
    coef_grad,mse,all_coef=LR.fit_autograd(X, y,20) # here you can use fit_non_vectorised / fit_autograd methods
    end_seconds_N=time.time()

    start_seconds_normal_N = time.time()
    coef_normal=LR.fit_normal(X, y) # here you can use fit_non_vectorised / fit_autograd methods
    end_seconds_normal_N=time.time()

    time_normal_N.append(end_seconds_normal_N-start_seconds_normal_N)
    time_grad_N.append(end_seconds_N-start_seconds_N)



print('with normal equation varying samples, time:',time_normal_N)
print('with gradient descent varying samples, time:',time_grad_N)
print('with normal equation varying features,time:',time_normal_P)
print('with gradient descent varying features, time:',time_grad_P)
N1=list(range(20,70,10))
P1=list(range(1,6))
plt.figure()
plt.scatter(P1,time_grad_P)
plt.xlabel('number of features')
plt.ylabel('time taken')
plt.title('For Gradient descent')
plt.figure()
plt.scatter(P1,time_normal_P)
plt.xlabel('number of features')
plt.ylabel('time taken')
plt.title('For normal equation')
plt.figure()
plt.scatter(N1,time_grad_N)
plt.xlabel('number of samples')
plt.ylabel('time taken')
plt.title('For gradient descent')
plt.figure()
plt.scatter(N1,time_normal_N)
plt.xlabel('number of samples')
plt.ylabel('time taken')
plt.title('For normal equation')


plt.show()