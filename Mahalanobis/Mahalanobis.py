import numpy as np
x=np.random.random(10)
y=np.random.random(10)

#马氏距离要求样本数要大于维数，否则无法求协方差矩阵
#此处进行转置，表示10个样本，每个样本2维
X=np.vstack([x,y])
print(X)
XT=X.T

#方法一：根据公式求解
S=np.cov(X)   #两个维度之间协方差矩阵
SI = np.linalg.inv(S) #协方差矩阵的逆矩阵
#马氏距离计算两个样本之间的距离，此处共有10个样本，两两组合，共有45个距离。
n=XT.shape[0]
d1=[]
for i in range(0,n):
    for j in range(i+1,n):
        delta=XT[i]-XT[j]
        d=np.sqrt(np.dot(np.dot(delta,SI),delta.T))
        d1.append(d)
print(d1)    
#方法二：根据scipy库求解
from scipy.spatial.distance import pdist
d2=pdist(XT,'mahalanobis')
print(d2)