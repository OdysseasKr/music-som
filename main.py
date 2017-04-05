import numpy as np
import time
import h5py
from som import SelfOrganisingMap

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

# Create input
input_file = h5py.File('ds.h5', 'r')

num_samples_per_class = 80
sample_dimension = input_file['vector_size'][0]

num_of_samples = (len(input_file.keys())-1) * num_samples_per_class
genres = input_file.keys()
genres.remove('vector_size')

train_labels = np.chararray((num_of_samples,), itemsize=10)
print(sample_dimension)
train_samples = np.empty((num_of_samples, sample_dimension), dtype=np.float32)

print("Preparing Data...")
train_index = 0
test_index = 0
for label in genres:
    for sample in input_file[label][:num_samples_per_class]:
        train_labels[train_index] = label
        train_samples[train_index] = np.array(sample)
        train_index += 1

somap = SelfOrganisingMap(4, sample_dimension, set(train_labels.tolist()))
train_samples, train_labels = unison_shuffled_copies(train_samples, train_labels)

print("Training Map...")
somap.train(train_samples[:21],train_labels[:21],1,3)
somap.train(train_samples[21:31],train_labels[21:31],0.5,3)
somap.train(train_samples[31:41],train_labels[31:41],0.1,1)
somap.train(train_samples[41:],train_labels[41:],0.1,0.1)

print(somap.sample_labels)
print(somap.Wl)
somap.plot_map_labels(['brown', 'yellow', 'cyan', 'magenta',
            'green','red', 'purple', 'blue'])

somap.export_map("map.h5")
