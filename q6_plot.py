import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing.polynomial_features import PolynomialFeatures
from linearRegression.linearRegression import LinearRegression



x = np.array([i*np.pi/180 for i in range(60,300,4)])
np.random.seed(10)  #Setting seed for reproducibility
y = 4*x + 7 + np.random.normal(0,3,len(x))


num_variations=np.floor(x.shape[0]/5)
num_variations=int(num_variations)
print(num_variations)
num_resi = x.shape[0]%5

degree = [1,3,5,7,9]
include_bias = True

theta_var=[]

for i in range(num_variations):
    theta_deg = []
    ind = np.random.choice(x.shape[0], 5 * (i + 1), replace=False)
    x_new = x[ind]
    x_new = pd.DataFrame(x_new)
    y_new = y[ind]
    y_new = pd.Series(y_new)

    for n in degree:
        poly = PolynomialFeatures(n,include_bias)
        X = poly.transform(x_new)
        L = LinearRegression(fit_intercept=include_bias)
        theta,mse, all_coef = L.fit_non_vectorised(X, y_new, X.shape[0], n_iter=5, lr=0.01, lr_type='constant')
        theta_deg.append(max(abs(theta)))
    theta_var.append(theta_deg)

if not(num_resi)==0:

    x_new=pd.DataFrame(x)
    y_new=pd.Series(y)

    for n in degree:
        poly = PolynomialFeatures(n,include_bias)
        X = poly.transform(x_new)
        L = LinearRegression(fit_intercept=include_bias)
        theta,mse, all_coef = L.fit_non_vectorised(X, y_new, X.shape[0], n_iter=10, lr=0.01, lr_type='constant')
        theta_deg.append(max(abs(theta)))
    theta_var.append(theta_deg)





for i in range(len(theta_var)):
    fig = plt.figure(i+1)
    plt.scatter(degree,theta_var[i])
    plt.title('N:{}'.format(i+1))
    plt.xlabel('degree')
    plt.ylabel('max(abs(theta))')
plt.show()