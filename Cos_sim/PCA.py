import numpy as np
import csv
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


tmp = np.loadtxt("Cos_Sim_Data_vector.csv", dtype=np.str, delimiter=",")
X = tmp[1:,2:].astype(np.float)
#print (data)
pca = PCA(n_components=2)
pca.fit(X)
result = pca.transform(X)
plt.scatter(result[:,1],result[:,0])
plt.show()

#fig = plt.figure()
#ax = fig.add_subplot(111,projection= '3d')
#ax.scatter(result[:,0],result[:,1],result[:,2])
#print (result,result[:, 1])
#plt.show()
