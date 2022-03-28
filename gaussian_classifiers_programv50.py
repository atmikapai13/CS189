# Import Relevant Libraries

import numpy as np
import scipy.cluster
import scipy.io
import scipy.ndimage
import matplotlib
import matplotlib.pyplot as plt
from gda import GDA
import pandas as pd
from scipy.stats import multivariate_normal
from matplotlib import interactive
import seaborn
import itertools


# Load Data and split it into Training and Validation Sets

def train_val_split(data, labels, val_size):
    num_items = len(data)
    assert num_items == len(labels)
    assert val_size >= 0
    if val_size < 1.0:
        val_size = int(num_items * val_size)
    train_size = num_items - val_size
    idx = np.random.permutation(num_items)
    data_train = data[idx][:train_size]
    label_train = labels[idx][:train_size]
    data_val = data[idx][train_size:]
    label_val = labels[idx][train_size:]
    return data_train, data_val, label_train, label_val

mnist = scipy.io.loadmat("mnist_data.mat")

train_data, train_labels = mnist['training_data'], mnist['training_labels']
train_data2, train_labels2 = mnist['training_data'], mnist['training_labels']
train_data_normalized = scipy.cluster.vq.whiten(train_data)
classes = np.unique(train_labels)
mnist_extended_dict = {}

print("Shape of data matrix:", train_data_normalized.shape)
print("Shape of labels vector:", train_labels.shape)
print("Number of classes:", len(classes))

for i in range(5):
    sample = train_data_normalized[i].reshape(28, 28)
    plt.figure()
    plt.imshow(sample)

#4a) Fitting a Gaussian Distribution to each class


def gaussian_digits(train_data, train_labels, classes):
    '''
    Fit a gaussian distribution to each digit class by computing the mean and covariance for each digit class. 
    
    Inputs:
        - train_data: n x d 
        - train_labels: n x 1
        - classes: 10
    
    Outputs:
        - mnist_fitted: Dictionary containing an entry per label that describes the label's mean and covariance
                        (i.e. mnist_fitted[8] = (mean_8, cov_8))
    '''
    ### YOUR CODE HERE ###
    mnist_fitted = {}
    #augmented_data = np.concatenate((train_data, train_labels), axis = 1)
    train_labels2 = np.squeeze(train_labels)
    size = train_labels2.shape[0]
    for i in classes:
        indices = np.where(train_labels2 == i)
        data = np.squeeze(np.take(train_data, indices, axis=0))
        mu = data.mean(axis=0)
        sigma = np.cov(data, rowvar=False) + np.identity(784)*(1**-5) #adding a small value 
        gaussian = multivariate_normal(mu, sigma)
        mnist_fitted[i] = [mu, sigma]
        mnist_extended_dict[i] = [mu, sigma, gaussian]
    return mnist_fitted

mnist_fitted = gaussian_digits(train_data_normalized, train_labels, classes)


#4b) Visualize the Covariance Matrix for a Particular Class

import seaborn as sns
#plt.matshow(class_1_cov)
class_1_cov = mnist_fitted.get(1)[1]
sns.set_theme(style="white")
cmap = sns.diverging_palette(230, 20, as_cmap=True)
ax = plt.axes()
cov_plot = sns.heatmap(class_1_cov, cmap=cmap, ax=ax)
ax.set_title("Covariance Matrix for Class 1")
plt.savefig('covariance_plot')

#4c) Gaussian Discriminant Analysis

# Split the data into training and validation and initialize hyperparamters

train_data, val_data, train_labels, val_labels = train_val_split(train_data_normalized, train_labels, val_size=10000)
val_labels.flatten()
num_training = [100, 200, 500, 1000, 2000, 5000, 10000, 30000, 50000]


def train_gda(train_data, train_labels, val_data, val_labels, num_training):
    '''
    For each number in num_training, fit a GDA model to the train set and evaluate its performance on 
    the validation set. Log the LDA error.
    
    Inputs:
        - train_data: (n-k) x d matrix
        - train_labels: (n-k) x 1 vector
        - val_data: k x d matrix
        - val_labels: k x 1 vector
        - num_training: List of different values that specify the number of train data points to use to fit the GDA model
        
    Outputs:
        - lda_errors: List of LDA errors (len(lda_errors) == len(num_training)))
        - returning y_pred of val set for Kaggle competition
    '''
    lda_errors = np.array([])
    val_pred = {}
    test_pred = {}

    def gda_func(data_point, mu, sigma, prior_prob):
        return mu.T@np.linalg.inv(sigma)@data_point - 1/2*mu.T@np.linalg.inv(sigma)@mu + np.log(prior_prob)
        
    
    for num in num_training:
        num_training_data = train_data[:num, :]
        num_training_labels = train_labels[:num, :]
        num_training_labels2 = np.squeeze(num_training_labels)
        
        uniques, counts = np.unique(num_training_labels2, return_counts=True)
        prior_probs = counts/len(num_training_labels2)

        num_mnist_fitted = gaussian_digits(num_training_data, num_training_labels, classes)

        class_cov = np.array([num_mnist_fitted.get(i)[1] for i in classes])
        inv_sigma = np.linalg.inv(np.mean(class_cov, axis=0))

        class_mu = np.array([num_mnist_fitted.get(i)[0] for i in classes])

        class_preds = np.ones(train_data.shape[0])
        for i in classes:
            mu_transpose = np.transpose(class_mu[i])
            intercept = -0.5*mu_transpose@inv_sigma@mu_transpose
            slope = mu_transpose@inv_sigma@train_data.T
            prob = np.log(prior_probs[i])
            score = intercept+slope+prob
            class_preds = np.vstack((class_preds, score))
                
        
        class_preds = np.delete(class_preds, 0, 0)
        num_test_predictions = class_preds.T.argmax(axis=1)
        test_pred[num] = num_test_predictions
                
        
        class_preds = np.ones(val_data.shape[0])
        for i in classes:
            mu_transpose = np.transpose(class_mu[i])
            intercept = -0.5*mu_transpose@inv_sigma@mu_transpose
            slope = mu_transpose@inv_sigma@val_data.T
            prob = np.log(prior_probs[i])
            score = intercept+slope+prob
            class_preds = np.vstack((class_preds, score))
                
        
        class_preds = np.delete(class_preds, 0, 0)
        num_val_predictions = class_preds.T.argmax(axis=1)
        val_pred[num] = num_val_predictions
        
        comparing = [(x == y == 1) for x, y in zip(num_val_predictions, val_labels)]
        comparing = np.squeeze(comparing)
        lda_err = 1 - np.true_divide(np.count_nonzero(comparing==True), len(comparing))
        lda_errors = np.append(lda_errors, lda_err)
        
        print("When fit on {:0.0f} training examples, LDA error is {:0.4f}".format(num, lda_err))
                                            
    return lda_errors, val_pred


lda_errors, val_pred = train_gda(train_data, train_labels, val_data, val_labels, num_training)

plt.figure(figsize=(8, 8))
plt.title("MNIST Training Error Rate Using LDA vs Number of Training Examples")
plt.xlabel("Number of Training Examples")
plt.ylabel("Training Error Rate")
plt.plot(num_training, lda_errors)
plt.savefig('lda_error')

lda_val_errors = {}
val_labels2 = np.squeeze(val_labels)

for i in classes:
    class_validation = []
    for key in num_training:
        preds = val_pred.get(key)
        indices = np.where(val_labels2 == i)
        total = np.count_nonzero(val_labels2 == i)
        y_pred_i = np.squeeze(np.take(preds, indices, axis=0))
        accurate_pred = np.count_nonzero(y_pred_i == i)
        score = accurate_pred/total
        class_validation = np.append(class_validation, score)
    
    lda_val_errors[i] = class_validation

fig, ax = plt.subplots(figsize=(7, 7))
ax.plot(num_training, lda_val_errors.get(0), label = "class 0")
ax.plot(num_training, lda_val_errors.get(1), label = "class 1")
ax.plot(num_training, lda_val_errors.get(2), label = "class 2")
ax.plot(num_training, lda_val_errors.get(3), label = "class 3")
ax.plot(num_training, lda_val_errors.get(4), label = "class 4")
ax.plot(num_training, lda_val_errors.get(5), label = "class 5")
ax.plot(num_training, lda_val_errors.get(6), label = "class 6")
ax.plot(num_training, lda_val_errors.get(7), label = "class 7")
ax.plot(num_training, lda_val_errors.get(8), label = "class 8")
ax.plot(num_training, lda_val_errors.get(9), label = "class 9")
ax.legend()
ax.set_xlabel('# of training plots')
ax.set_ylabel('Error Rate')
ax.set_title('LDA Classification, digitwise')
fig.savefig('digitwise_lda.png')


#For Kaggle

test_data = mnist['test_data']
fake_kaggle_error, test_pred_kaggle = train_gda(train_data, train_labels, test_data, np.ones(10000), [50000])
 
def results_to_csv(y_test):
    y_test = y_test.astype(int)
    df = pd.DataFrame({'Category': y_test})
    df.index += 1 #Ensures that the index starts at 1. 
    df.to_csv('submission.csv', index_label='Id')

results_to_csv(test_pred_kaggle.get(50000))

#Kaggle submission under Atmika Pai
#Model Accuracy: 81.7%
    

