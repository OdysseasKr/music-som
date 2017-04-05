# music-som
Kohonen self organising maps for music clustering in Python

This repo contains a general implementation of Self Organising Maps in Python.

This repo also contains scripts for experimenting with the GTZAN Dataset containing 30-second music tracks categorised by genre. Download the dataset [here](http://marsyas.info/downloads/datasets.html).


<p align="center">
<img src="https://raw.githubusercontent.com/OdysseasKr/music-som/master/figure_1.png" alt="demo figure" width="533" height="400">
</p>


## Using the dataset
To use the dataset, extract the files and use the createh5.py file to extract features using Librosa and create a .h5 file containing the training set. Usage:
```bash
python createh5.py /path/to/GTZAN filename.h5
```

## Using the SelfOrganisingMap class
The `SelfOrganisingMap` class has been implemented as a __general__ class to use in your Python experiments independantly from your data. Simply instantiate an object passing the dimension of the map and the dimension of the feature vector.

E.g. using MNIST dataset on a 16x16 Kohonen map
```python
from som import SelfOrganisingMap
map = SelfOrgansingMap(16, 784)
```

The `SelfOrganisingMap` class is using the Gaussian function to scale neighbors.

For more info on how Kohonen Self Organising Maps work, refer to this short [tutorial](http://davis.wpi.edu/~matt/courses/soms/).
