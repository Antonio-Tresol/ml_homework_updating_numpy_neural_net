# config_regression.py
from data.dataset import *
import numpy as np

# Hyperparameters
batch_size = 10
validate_every_no_of_batches = 50
epochs = 1000
input_size = 1
output_size = 1
hidden_shapes = [10, 5]  # Adjust hidden layer sizes as needed
lr = 0.01
has_dropout = False
dropout_perc = 0.5
output_log = "runs/regression_log.txt"

# Create a simple regression dataset (y = 2x + 1)
x = np.array([[i] for i in range(1000)])
y = 2 * x + 1
data = dataset(x, y, batch_size)

# Split the dataset
splitter = dataset_splitter(data.compl_x, data.compl_y, batch_size, 0.8, 0.2)
ds_train = splitter.ds_train
ds_val = splitter.ds_val
ds_test = splitter.ds_test
