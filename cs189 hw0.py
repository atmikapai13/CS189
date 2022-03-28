#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal #https://stackoverflow.com/questions/11615664/multivariate-normal-density-in-python


# In[3]:


#source: https://cogmaster-stats.github.io/python-cogstats/auto_examples/matplotlib_demos/plot_contour.html

#5a
mu = [1, 1]
sigma = [[1, 0], [0, 2]]

x = np.linspace(-3, 5, 100)
y = np.linspace(-3, 5, 100)
X,Y = np.meshgrid(x, y) 
depth = np.dstack((X, Y)) 
Z = multivariate_normal(mu, sigma).pdf(depth)
plt.contourf(X, Y, Z, 8, alpha=.75, cmap=plt.cm.hot);


# In[4]:


#5b
mu = [-1, 2]
sigma = [[2, 1], [1, 4]]

x = np.linspace(-5, 4, 100)
y = np.linspace(-5, 8, 100)
X,Y = np.meshgrid(x, y)
depth = np.dstack((X, Y))
Z = multivariate_normal(mu, sigma).pdf(depth)
plt.contourf(X, Y, Z, 8, alpha=.75, cmap=plt.cm.hot);


# In[5]:


#5c
mu1 = [0, 2]
mu2 = [2, 0]
sigma = [[2, 1], [1, 1]]

x = np.linspace(-5, 7, 100)
y = np.linspace(-3, 5, 100)
X,Y = np.meshgrid(x, y)
depth = np.dstack((X, Y))
Z = multivariate_normal(mu1, sigma).pdf(depth) - multivariate_normal(mu2, sigma).pdf(depth)
plt.contourf(X, Y, Z, 8, alpha=.75, cmap=plt.cm.hot);


# In[6]:


#5d
mu1 = [0, 2]
mu2 = [2, 0]
sigma1 = [[2, 1], [1, 1]]
sigma2 = [[2, 1], [1, 4]]

x = np.linspace(-5, 7, 100)
y = np.linspace(-3, 5, 100)
X,Y = np.meshgrid(x, y)
depth = np.dstack((X, Y))
Z = multivariate_normal(mu1, sigma1).pdf(depth) - multivariate_normal(mu2, sigma2).pdf(depth)
plt.contourf(X, Y, Z, 8, alpha=.75, cmap=plt.cm.hot);


# In[7]:


#53
mu1 = [1, 1]
mu2 = [-1, -1]
sigma1 = [[2, 0], [0, 1]]
sigma2 = [[2, 1], [1, 2]]

x = np.linspace(-5, 7, 100)
y = np.linspace(-5, 3, 100)
X,Y = np.meshgrid(x, y)
depth = np.dstack((X, Y))
Z = multivariate_normal(mu1, sigma1).pdf(depth) - multivariate_normal(mu2, sigma2).pdf(depth)
plt.contourf(X, Y, Z, 8, alpha=.75, cmap=plt.cm.hot);


# In[8]:


#Question 6
np.random.seed(500)
x1 = np.random.normal(3, np.sqrt(9), 100)
x2 = 0.5*x1 + np.random.normal(4, np.sqrt(4), 100)


# In[9]:


#6a
#mean of sample
sample1 = np.dstack((x1, x2))
sample1_mean = np.mean(sample1, axis = 1)
lower_mean = sample1_mean[0][0]
upper_mean = sample1_mean[0][1]
print("(" + str(np.round(lower_mean, 5)) + " , " + str(np.round(upper_mean, 5)) + ")") 


# In[10]:


#6b
#2 x 2 covariance matrix of sample
np.cov(x1, x2)


# In[11]:


#6c
#eigenvalues and eigenvectors of 2 x 2 covariance matrix of sample
eigenvalue, eigenvector = np.linalg.eig(np.cov(x1, x2))
val0, vector0 = eigenvalue[0], eigenvector[0]
val1, vector1 = eigenvalue[1], eigenvector[1]
print(str(np.round(val0, 5)) + " eigenvalue with eigenvector:" + str(vector0))

print(str(np.round(val1, 5)) + " eigenvalue with eigenvector:" + str(vector1))   
      


# In[12]:


#6d
ax1 = plt.gca() #you first need to get the axis handle
ax1.set_aspect(1)
plt.scatter(x1, x2);
plt.xlim(-15, 15);
plt.ylim(-15, 15);
plt.xlabel('X1')
plt.ylabel('X2')
plt.arrow(lower_mean, upper_mean, vector0[0]*val0, vector1[0]*val0, shape='full', width=0.25);
plt.arrow(lower_mean, upper_mean, vector0[1]*val1, vector1[1]*val1, shape='full', width=0.25);
plt.title('X1 and X2 and respective covariance eigenvectors');


# In[13]:


#6e
v1 = max(val0, val1)
v1_vector = vector0
rotation_of_centered_x = eigenvector.T @ (sample1[0] - sample1_mean[0]).T
ax2 = plt.gca() #you first need to get the axis handle
ax2.set_aspect(1)
plt.scatter(rotation_of_centered_x[0], rotation_of_centered_x[1]);
plt.xlim(-15, 15);
plt.ylim(-15, 15);
plt.xlabel('X1');
plt.ylabel('X2');
plt.title('X1 and X2 rotated to coordinate space aligning with eigenvectors');

