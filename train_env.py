#Main Script

import matplotlib.pyplot as plt
import lstm
import numpy as np

def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()
    
    
seq_len = 10
train_samples = 100000
test_samples = 2000

x_raw, y_raw, info = lstm.load_data(
    path="../2014-04-01_1m_172800.csv", 
    sequence_length = seq_len, 
    row_start_ind=0, 
    in_column_ind=[0,1,2,3,4,5,6],
    out_column_ind=[7,8,9,10,11,12], 
    do_normalize=True)

#print(info)

x_dim = x_raw.shape[2]
y_dim = y_raw.shape[2]
x_train, y_train = x_raw[:train_samples,:,:], y_raw[:train_samples,:,:]
x_test, y_test = x_raw[-test_samples:,:,:], y_raw[-test_samples:,:,:]

m_ = lstm.build_model(1, seq_len, x_dim, 100, 1, y_dim, False)
#m_.load_weights("./save_model/env.h5")
m_.fit(x_train, y_train, batch_size=1, nb_epoch=10)
m_.save_weights("./save_model/env.h5")
y_pred = lstm.predict_sequence(m_, x_test, batch_size=1)

for i in range(y_dim):
    plot_results(y_pred.reshape(-1, y_dim).transpose()[i], y_test.reshape(-1, y_dim).transpose()[i])
