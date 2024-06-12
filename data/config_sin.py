from data.dataset import *
import numpy as np

#hyperparameters
batch_size = 16
validate_every_no_of_batches = 8
epochs = 1000
input_size = 1
output_size = 1
hidden_shapes = [10,10,10]
lr = 0.085
has_dropout=False
dropout_perc=0.5
output_log = "runs/sin_log.txt"

# Generate a sine function dataset
x_values = np.linspace(-np.pi, np.pi, 200).reshape(-1, 1)  # 200 points between -pi and pi
y_values = np.sin(x_values).reshape(-1, 1)  # Sine values

# Combine x and y values into a single array for shuffling
data = np.hstack((x_values, y_values))

# Shuffle the dataset
np.random.shuffle(data)

# Split the dataset (80% train, 20% test)
split_idx = int(len(data) * 0.8)
train_data = data[:split_idx]
test_data = data[split_idx:]

# Extract inputs and targets for training and testing sets
x_train, y_train = train_data[:, 0].reshape(-1, 1), train_data[:, 1].reshape(-1, 1)
x_test, y_test = test_data[:, 0].reshape(-1, 1), test_data[:, 1].reshape(-1, 1)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
#Sin dataset - baseline to check backprop, update and forward calculations
ds_train = dataset(x_train, y_train, batch_size)
ds_test = dataset(x_test, y_test, batch_size)
ds_val = dataset(x_test, y_test, batch_size)
