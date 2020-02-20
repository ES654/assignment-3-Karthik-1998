import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from preprocessing.polynomial_features import PolynomialFeatures
from linearRegression.linearRegression import LinearRegression

x = np.array([i*np.pi/180 for i in range(60,300,4)])
np.random.seed(10)  #Setting seed for reproducibility
y = 4*x + 7 + np.random.normal(0,3,len(x))

x=pd.DataFrame(x)
y=pd.Series(y)

max_degree = 10
include_bias = True
theta_deg=[]

for n in range(max_degree):
    poly = PolynomialFeatures(n+1,include_bias)

    X = poly.transform(x)
    L = LinearRegression(fit_intercept=include_bias)
    theta,mse, all_coef = L.fit_non_vectorised(X, y, X.shape[0], n_iter=5, lr=0.01, lr_type='constant')
    theta_deg.append(max(abs(theta)))

print(theta_deg)
print(x.shape)

plt.scatter(list(range(1,max_degree+1)),theta_deg)
plt.xlabel('degree')
plt.ylabel('max(theta)')
plt.title('max(abs(theta)) vs degree')
plt.show()

