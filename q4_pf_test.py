import numpy as np
from preprocessing.polynomial_features import PolynomialFeatures
import pandas as pd


include_bias=True
np.random.seed(42)

N = 30
P = 2
X = np.array([1,2])
#X = pd.DataFrame(np.random.randn(N, P))
poly = PolynomialFeatures(2,include_bias)
x = poly.transform(X)
print(x)
