from sklearn.preprocessing import StandardScaler
import numpy as np

x = np.array([[10001, 2, 55], [16020, 4, 11], [13131, 8, 22]])
X_scaler = StandardScaler()
x = X_scaler.fit_transform(x)
# print (x)

m = 3  # sample number
#cov_mat = np.dot(x.transpose(), x) / (m - 1)
cov_mat = np.cov(x, rowvar=0)
print(cov_mat)
