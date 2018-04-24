import numpy as np
from sklearn.linear_model import LinearRegression
x = 2 * np.random.rand(100,1)
y = 4 +3 * x + np.random.randn(100,1)
#Linear using sklearn
lin_reg = LinearRegression()
lin_reg.fit(x, y)
