###################################################
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
###################################################



def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

PATH = 'C:/Users/lg/PycharmProjects/test/cifar10/data/'
FILE_LIST = os.listdir(PATH)




##################  data loading   ####################

dt = list()
label = list()
label_key = b'labels'
dt_key = b'data'

for i, FILE in enumerate(FILE_LIST):
    unpic = unpickle(PATH+FILE)
    tmp_label = unpic[label_key]
    tmp_dt = unpic[dt_key]

    if(i == 0):
        dt = tmp_dt
        label = tmp_label
    else:
        dt = np.concatenate((dt, tmp_dt), axis=0) #ndarray merge
        label = label + tmp_label


len(dt)
len(label)



################# One-Hot Encoding #######################
enc = OneHotEncoder()

label = np.reshape(np.asarray(label), (-1, 1))
enc.fit(label)
label = enc.transform(label).toarray()
label



#################  Normalization  #######################
def MinMaxSclaer(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    return numerator/(denominator + 1e-7)


xdt = [0.0]*len(dt)
for j in range(0, len(dt)):
    x = dt[j]
    x2 = MinMaxSclaer(x)
    xdt[j] = x2


np.shape(xdt)    #x_shape : (60000,3072)
np.shape(label)  #y_shape : (60000,10)



##################  data partitioning  ####################
train_xdt = xdt[:50000][:]
train_ydt = label[:50000]

test_xdt = xdt[50000:][:]
test_ydt = label[50000:]

print("Handling.py bas been completed!")