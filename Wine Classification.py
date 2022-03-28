#!/usr/bin/env python
# coding: utf-8

# # Q3: Wally's Winery
# 
# CS 189 Spring 2022
# 
# Author: Atmika Pai
# 
# Wally Walter and the Walter family run the most popular winery in Napa. This summer, UC Berkeley students are flooding in to try the Walter Family's fine wines! Wally's children, Wilma and Willy, are slowly helping Wally take ownership of the family business. However, they do not possess the quintessential Walter talent: identifying red and white wines based on attributes! To help them, Wally has decided to train a machine learning classifier to distinguish Walter's Red Wine from Walter's White Wine.
# 
# In this exercise, we will build a logistic regression classifier to classify whether a specific wine is a white (class label 0) or red (class label 1) wine, based on its features
# 
# Use of automatic logistic regression libraries/packages is prohibited for this question. If you
# are coding in python, it is better to use np.true_divide for evaluating logistic functions
# as its code is numerically stable, and doesn’t produce NaN or MathOverflow exceptions.
# 
# 
# <div>
#     
# <img src="wine.jpg" width="500"/>
# </div>

# In[1]:


# Import Relevant Libraries

import scipy
from scipy import io
import numpy as np
import matplotlib.pyplot as plt
import time


# In[2]:


# Load Data

data = scipy.io.loadmat('wine.mat')
features = data['X']
labels = data['y']
num_features=len(features[0])
num_examples = len(features)

print("Wine data currently has {} features and {} data points".format(num_features, num_examples))


# ## Q3.1 Preprocessing the Data
# 
# Now that we have loaded the data, we are ready to preprocess it. Preprocessing comes in three steps:
# 
# 1) Append an extra feature and label to the data points
# 
# 2) Split the data into training and test sets
# 
# 3) Normalize the features
# 
# Notice that we split the data into training and test sets <b> before </b> we normalize the features. If we normalized the features and then split the data into different sets, then the mean and variance of the features is no longer normalized per dataset.

# In[3]:


# Q3.1 

def append_feature(features, labels, num_features, num_examples):
    
    '''
    Append an extra feature to the data set.
    Then, append the labels to the data set.
    The resulting data matrix should look like this: [X', y]
    
    Where X' is feature matrix with an additional feature, and y are the labels
    
    Inputs: 
        - features: n x d
        - labels: n x 1
        - num_features: n
        - num_examples: d
    
    Output:
        - augmented data matrix: n x (d + 2)
        - new num_features: n + 1
    '''
    extra_features = np.insert(features, num_features, 1, axis = 1)
    labels_arr = [item for sublist in labels for item in sublist]
    augmented_data_matrix = np.insert(extra_features, num_features+1, labels_arr, axis = 1)
    return augmented_data_matrix, num_features + 1

def split_data(augmented_data, val_size=1000):
    '''
    Split the data into training and validation sets
    
    Input: 
        - augmented data matrix: n x (d + 1)
        - val_size: k
    
    Output: 
        - Training Set: (n - k) x (d + 1)
        - Validation Set: k x (d + 1)
        
    '''
    ### YOUR CODE HERE ###
    
    num_examples = augmented_data.shape[0]
    msk = np.full((num_examples, 1), True, dtype=bool)
    indices = np.random.choice(np.arange(num_examples), replace=False, size=val_size)
    msk[indices] = False
    msk = np.concatenate(np.transpose(msk), axis=0)
    train = augmented_data[msk]
    validate = augmented_data[~msk]
    
    #return augmented_data[val_size:augmented_data.shape[0], :], augmented_data[0:val_size, :]
    return train, validate

def normalize_features(train_set, val_set, num_features):
    '''
    Normalize the data by shifting it to 0 mean and rescaling it to have variance 1
    
    Input: 
        - Training Set: (n - k) x (d + 1)
        - Validation Set: k x (d + 1)
        - num_featres: d
    
    Output:
        - Normalized Training Set: (n - k) x (d + 1)
        - Normalized Validation Set: k x (d + 1)
        
    normalizing: (normed - normed.mean(axis=0))/normed.std(axis=0)
    '''
    def column(matrix, i):
        return [row[i] for row in matrix]
    
    ### YOUR CODE HERE ###
    def normalizer(df, feat):
        num_examples = df.shape[0]
        rel_df = df[:, 0:feat]
        normed = np.true_divide((rel_df - rel_df.mean(axis=0)), rel_df.std(axis=0))
        return normed
    
    normed_train_set = normalizer(train_set, total_features-1)
    normed_val_set = normalizer(val_set, total_features-1)
    
    normed_train_set_full = np.concatenate((normed_train_set, train_set[:, 12:]), axis=1)
    normed_val_set_full = np.concatenate((normed_val_set, val_set[:, 12:]), axis=1)
    
    return normed_train_set_full, normed_val_set_full


# In[4]:


val_size = 1000
train_size = num_examples - val_size

augmented_data, total_features = append_feature(features, labels, num_features, num_examples)
train_set, val_set = split_data(augmented_data, val_size)
train_set, val_set = normalize_features(train_set, val_set, num_features)


# In[5]:


import pandas as pd
print("Train_set:")
display(pd.DataFrame(train_set).head())
print("Val_set:")
display(pd.DataFrame(val_set).head())      


# ## Q3.3 Training Your Classifier
# 
# Now it is time to build and train your classifier! Every machine learning pipeline consists of 4 components:
# 
# 1) Model/Classifier – In this case, Logistic Regression
# 
# 2) Optimizer – In this case, Batch Gradient Descent
# 
# 3) Loss – In this case, log-likelihood using binary cross entropy
# 
# 4) Data – Wine Data
# 
# In this exercise, our model is a logistic regression model. We will begin by optimizing it with batch gradient descent, and the loss is the log of the binary cross entropy. In batch gradient descent, we calculate the gradient at each time step using all of the data points in the training set. 
# 
# ### Your tasks are as follows:
# 
# 1) Implement the Logistic Regression Function
# 
# 2) Implement training with Batch Gradient Descent
# 
# 
# ### To train the model with batch gradient descent, you must:
# 
# 1) Commpute the predictions from the classifier 
# 
# 2) Compute the loss (negative of the likelihood)
# 
# 3) Compute the gradient of the loss with respect to weight vector w
# 
# 4) Perform the gradient descent step
# 
# 5) Loop through steps 1-4 num_iters times

# In[6]:


# Logistic Regression function
def sigmoid(w, X):
    '''
    Implement the logistic regression function
    
    Inputs:
        - w: d x 1 weight vector
        - X: n x d feature matrix
    
    Outputs:
        - s: n x 1 output of sigmoid
    '''
    
    return 1 / (1 + np.exp(-X@w))
    
# Q3.2

def batch_train(train_set, total_features, num_iter, lr=0.0001, reg_const=0.1):
    '''
    Learn a weight vector for the logistic regression classifier by iterating over the data
    and updating the weights using batch gradient descent
    
    Inputs: 
        - train set: n x (d + 1)
        - total features: d
        - # of training iterations
        - learning rate
        - l2 regularization constant
    
    Outputs: 
        - w: d x 1 weight vector
        - list of losses: # iters x 1 vector
    '''
    
    # We want to keep track of the loss per iteration so that we can plot it later
    loss = np.zeros((num_iter+1,))
    size = train_set.shape[0]
    X = train_set[:, 0:total_features]
    
    y = [data_point[-1] for data_point in train_set]
    
    
    # Initialize variables
    w = np.zeros((total_features,))
    grad = np.zeros((total_features,))
    ones = np.ones(num_examples)
    
    ### YOUR CODE HERE ###

    s = sigmoid(w, X)
    loss[0] = -1/size * np.sum(y * np.log(s) + (1 - np.mean(y)) * np.log(1-s) - 1/2*reg_const*np.linalg.norm(w)**2)
    
    for i in np.arange(num_iter-1):
        grad = 1/size * ((X.T @ (s - y)) + reg_const*w)
        w = w - lr * grad
        s = sigmoid(w, X)
        
        loss[i+1] = -1/size * np.sum(y * np.log(s) + (1 - np.mean(y)) * np.log(1-s) - 1/2*reg_const*np.linalg.norm(w)**2)
        
        if i % 500 == 0:
            print("Loss at Iteration {} is {}:".format(i, loss[i+1]))
    return w, loss

def calc_time(start_time, end_time):
    '''
    Prints the time that a process takes to complete in a readable manner
    '''
    hours, rem = divmod(end_time-start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Training completed with elapsed time {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))


# In[7]:


num_iter = 7000

start_time = time.time()

w, loss = batch_train(train_set, total_features, num_iter)

end_time = time.time()
calc_time(start_time, end_time)


# In[8]:


# Plotting cost vs. iterations
plt.plot(np.arange(num_iter+1), loss)
plt.xlabel('Number of iterations in training')
plt.ylabel('Cost at the end of training')
plt.title('Training loss vs. Number of iterations for Batch Gradient Descent')
plt.savefig('Wine_GD.png')
plt.show()

# Checking on validation set
s_test = sigmoid(w,val_set[:,:total_features])
diffe = np.rint(s_test)-val_set[:,total_features]
accuracy = (np.true_divide(diffe.size-np.count_nonzero(diffe),val_size))*100
print(np.rint(s_test).sum())
print("Validation Accuracy is %.2f%%" % (accuracy))


# ### You should expect to see the loss decrease significantly in the first 1000 iterations. The validation accuracy should be greater than 99%

# ## Q3.5 Training Your Classifier using Stochastic Gradient Descent
# 
# Now, we will try to train the classifier using stochastic gradient descent. In stochastic gradient descent, we calculate the gradient at each time step using a single, randomly selected data point. Because we are calculating the gradient from one sample per iteration, stochastic gradient descent serves only as an approximation of batch gradient descent. However, it is much faster to compute, because we do not need to use the whole training set to compute the gradient.
# 
# Implement training with stochastic gradient descent below.
# 
# <b> Note: Because SGD only computes the gradient from one sample per iteration, the magnitude of an SGD gradient will be n times smaller than the magnitude of a batch gradient, where n is the number of data points. To mitigate this issue, make sure to multiply the appropriate part of the gradient by n. </b> 

# In[9]:


# Q3.5
import random
def stochastic_train(train_set, total_features, train_size, num_iter, lr=1e-6, reg_const=0.1, decay=False):
    '''
    Learn a weight vector for the logistic regression classifier by iterating over the data
    and updating the weights using stochastic gradient descent
    
    Inputs: 
        - train set: n x (d + 1)
        - total features: d
        - train_size: n
        - # of training iterations
        - learning rate
        - l2 regularization constant
        - decaying lr: boolean to determine whether or not we are decaying the learning rate
    
    Outputs: 
        - w: d x 1 weight vector
        - list of losses: # iters x 1 vector
    '''
    
    # We want to keep track of the loss per iteration so that we can plot it later
    loss = np.zeros((num_iter+1,))
    
    # Initialize variables
    w = np.zeros((total_features,))
    grad = np.zeros((total_features,))
    size = train_set.shape[0]
    
    X = train_set[:, 0:total_features]
    y = [item[-1] for item in train_set]
    
    ### YOUR CODE HERE ###
    s = sigmoid(w, X)
    loss[0] = -1/size * np.sum(y * np.log(s) + (1 - np.mean(y)) * np.log(1-s) - 1/2*reg_const*np.linalg.norm(w)**2)
    
    original_lr = lr
    
    for i in np.arange(num_iter):
        
        if decay:
            lr = lr/(i+1)
            
        sample_index = random.randint(0, size-1)
        x_batch, y_batch = train_set[sample_index][0:13], train_set[sample_index][-1]
        grad =  (size*(x_batch.T * (s[sample_index] - y_batch)) + reg_const*w)
        w = w-lr*grad
        s = sigmoid(w, X)

        loss[i+1]= -np.sum(y_batch * np.log(s[sample_index]) + (1 - np.mean(y_batch)) * np.log(1-s[sample_index]) - 1/2*reg_const*sum(w**2))
        
        if i % 500 == 0:
            print("Loss at Iteration {} is {}:".format(i, loss[i+1]))
            
    return w, loss


# In[212]:


start_time = time.time()

w, loss = stochastic_train(train_set, total_features, train_size, num_iter, lr=1e-6)

end_time = time.time()
calc_time(start_time, end_time)


# In[226]:


# Plotting cost vs. iterations
plt.plot(np.arange(num_iter+1), loss)
plt.xlabel('Number of iterations in training')
plt.ylabel('Cost at the end of training')
plt.title('Training loss vs. Number of iterations for Stochastic Gradient Descent')
plt.savefig('Wine_SGD.png')
plt.show()

# Checking on validation set
s_test = sigmoid(w,val_set[:,:total_features])
diffe = np.rint(s_test)-val_set[:,total_features]
accuracy = (np.true_divide(diffe.size-np.count_nonzero(diffe),val_size))*100
print(np.rint(s_test).sum())
print("SGD Validation Accuracy is %.2f%%" % (accuracy))


# ## Q 3.6 SGD with a Decaying Learning Rate
# 
# Instead of using a constant step size (learning rate) in SGD, you could use a step size that
# slowly shrinks from iteration to iteration. Run your SGD algorithm from question 3.4 with a learning rate that decays in every iteration. The learning rate in every iteration should decay as $lr = \frac{lr_0}{i + 1}$, where $lr_0$ was the original learning rate. Begin with a learning rate of 1e-4, and report your results.

# In[230]:


start_time = time.time()
decay_w, decay_loss = stochastic_train(train_set, total_features, train_size, num_iter, lr=1e-4, decay=True)
end_time = time.time()

calc_time(start_time, end_time)


# In[232]:


# Plotting cost vs. iterations
plt.plot(np.arange(num_iter+1), loss, "-r", label="no decay")
plt.plot(np.arange(num_iter+1),decay_loss, "-b", label="decay")
plt.xlabel('Number of iterations in training')
plt.ylabel('Cost at the end of training')
plt.legend(loc="upper left")
plt.title('SGD Training loss vs. iterations with decaying & const learning rate')
plt.savefig('Wine_SGD_combined.png')
plt.clf()
plt.close()

# Checking on validation set
s_test=sigmoid(w,val_set[:,:total_features])
ss_test=sigmoid(decay_w,val_set[:,:total_features])
diffe=np.rint(s_test)-val_set[:,total_features]
diffe_s=np.rint(ss_test)-val_set[:,total_features]
accuracy=(np.true_divide(diffe.size-np.count_nonzero(diffe),val_size))*100
accuracy_s=(np.true_divide(diffe_s.size-np.count_nonzero(diffe_s),val_size))*100
print("SGD Validation Accuracy (constant learning rate) is %.2f%%" % (accuracy))
print("SGD Validation Accuracy (decaying learning rate) is %.2f%%" % (accuracy_s))


# ## Great Job Finishing! Here are some Key Takeaways from this Exercise:
# 
# 1) Logistic Regression is a classifier that can predict between two classes. It can be iteratively optimized via gradient descent
# 
# 2) Stochastic Gradient Descent runs computationally quicker than Batch Gradient Descent, at the cost of some model performance
# 
# 3) One strategy to improve model training is to decay the learning rate of the optimizer over time. While this can lead to good results, one must be tricky about setting the decay rate and the initial learning rate.

# Wally is infinitely grateful for your help! Now that he has a wine classifier, he can pass down the family business in peace :)

# In[ ]:




