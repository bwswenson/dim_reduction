import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import MDS
plt.style.use('seaborn-whitegrid')

eps = 0.00
n = 100
t = np.linspace(0, 2*np.pi, n)
x1 = np.sin(t)
x2 = np.cos(t)
X = np.vstack((x1, x2)).transpose() + eps*np.random.rand(n, 2)
X_more1 = np.zeros((0, 2)) 
X_more2 = np.zeros((0, 2)) 
X_more1[:,0] = .1
X_more2[:,0] = -.1
X = np.vstack((X, X_more1, X_more2))


embedding = MDS(n_components=1)
X_low = embedding.fit_transform(X)
X_low = X_low.squeeze()
 
plt.figure(1).clf()
fig1, ax1 = plt.subplots(1,1, num=1)
ax1.plot(X[:,0], X[:,1], 'o', color='black')

plt.figure(2).clf()
fig2, ax2 = plt.subplots(1,1, num=2)
dummy = np.zeros_like(X_low)
ax2.plot(X_low, np.zeros_like(X_low), 'o', color='black')

fig3 = plt.figure()
ax3 = fig3.add_subplot(projection='3d')
ax3.scatter(X[:,0], X[:,1], X_low, marker='o')