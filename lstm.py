#Modeling Environment

import os
import time
import warnings
import numpy as np

#scikit-learn
from sklearn import preprocessing

#keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Reshape
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
warnings.filterwarnings("ignore") #Hide messy Numpy warnings

def load_data(path, sequence_length, row_start_ind, in_column_ind, out_column_ind, do_normalize):
    with open(path) as f:
        content = f.readlines()
    content = [a[0].split(",")for a in [x.strip().split("\n") for x in content]]
    np_content = np.array(content)
    np_content = np.asarray(np_content[row_start_ind:,in_column_ind+out_column_ind], dtype=np.float32)
    #normalize
    if do_normalize:
        np_content, info = normalize(np_content)
    else:
        info = None
    #manipulating
    x_temp = []
    y_temp = []
    for i in range(np_content.shape[0] - sequence_length):
        x_temp.append(np_content[i:i+sequence_length,:len(in_column_ind)])
        y_temp.append(np_content[i+sequence_length:i+sequence_length+1,len(in_column_ind):])
    return np.array(x_temp), np.array(y_temp), info

def normalize(data):
    m = np.mean(data, axis=0)
    s = np.std(data, axis=0)
    i = [m, s]
    d = preprocessing.scale(data)
    return d, i

def build_model(batch_size, sequence_length, x_dim, h_dim, num_hid_lay, y_dim):
    model = Sequential()
    model.add(LSTM(output_dim=h_dim, return_sequences=True, stateful=True, init='uniform', 
                   batch_input_shape=(batch_size, sequence_length, x_dim)))
    for i in range(num_hid_lay):
        model.add(LSTM(output_dim=h_dim, return_sequences=True, stateful=True, init='uniform'))
    model.add(Reshape((1, -1)))
    model.add(TimeDistributed(Dense(y_dim)))
    model.add(Activation('linear'))
    start = time.time()
    model.compile(loss='mean_squared_error', optimizer='rmsprop')
    #sgd = SGD(lr=0.1, decay=1e-4, momentum=0.2, nesterov=True)
    #model.compile(loss='mean_squared_error', optimizer=sgd)
    print("> Compilation Time : ", time.time() - start)
    return model

def predict_sequence(model, x_test, batch_size):
    pred = model.predict(x_test, batch_size=batch_size)
    return pred