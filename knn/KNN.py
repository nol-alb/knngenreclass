import numpy as np
import scipy as sp

### Helper Functions
def euc(vector, matrix):
    d = np.sum((vector - matrix)**2, axis = 1, keepdims = True)
    return np.sqrt(d.T)
def pairwise_distance(test_data, train_data):
    d = np.zeros((test_data.shape[1], train_data.shape[1]))
    for i,v in enumerate(test_data.T):
        d[i,:] = euc(v,train_data.T)
    return d
    
def infer_class (closeness, data_labels, classes):
    
    # first, try to do a simple majority vote
    #res_class = sp.stats.mode(class_labels, axis = 0)
    hist = np.histogram(data_labels, bins = len(classes), range = (min(classes), 
max(classes)))
 
    # check if we have a clear majority
    s = np.flip(np.sort(hist[0]))
    # if not, do a histgram weighted with the closeness
    if s[0] == s[1]:
        hist = np.histogram(data_labels, bins = len(classes), range = 
(min(classes), max(classes)), weights = closeness)
 
    return classes[np.argmax(hist[0])]


def knearestneighbor(test_data, train_data, train_label, k=1):
    # sanity checks
    if test_data.shape[0] != train_data.shape[0]:
        return -1
    # get dimensions
    num_features = test_data.shape[0]
    classes = np.unique(train_label).astype(int)
    # init result
    est_class = -1*np.ones(test_data.shape[1])
    # compute pairwise distances between all test_data and all train_data points
    # you may also use sp.spatial.distance.cdist
    d = pairwise_distance(test_data, train_data)
    # find lowest distances
    ind = np.argsort(d, axis = 1).astype(int)
    ind = ind[:,range(k)]
    #convert distance to closeness (easier later with the histogram)
    ma = np.amax(d)
    if ma <= 0:
        ma = 1
    d = 1-d/ma
    # infer which class
    for obs,index in enumerate(ind):
        est_class[obs] = infer_class (d[obs,index], train_label[index], classes)
    return np.squeeze(est_class.astype(int))