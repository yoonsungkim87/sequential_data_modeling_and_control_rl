import matplotlib.pyplot as plt
import lstm

def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()

seq_len = 10
train_samples = 10000
test_samples = 100
    
x_raw, y_raw, info = lstm.load_data(path="./data.csv", sequence_length = seq_len, row_start=2, column_start=2, output_border_start=55, do_normalize=True)

#print(x_raw.shape, y_raw.shape)

x_dim = x_raw.shape[2]
y_dim = y_raw.shape[2]
x_train, y_train = x_raw[:train_samples,:,:], y_raw[:train_samples,:,:]
x_test, y_test = x_raw[-test_samples:,:,:], y_raw[-test_samples:,:,:]

m_ = lstm.build_model(1, seq_len, x_dim, 100, 1, y_dim)
m_.fit(x_train, y_train, batch_size=1, nb_epoch=10)
y_pred = lstm.predict_sequence(m_, x_test, batch_size=1)

for i in range(y_dim):
    plot_results(y_pred.reshape(-1, y_dim).transpose()[i], y_test.reshape(-1, y_dim).transpose()[i])