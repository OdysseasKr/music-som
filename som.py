import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class SelfOrganisingMap:
    """
    Represents a self organising map
    """

    def __init__(self, map_dimension, sample_dimension, sample_labels=None):
        """
        map_dimension : The dimension of the map. The final map will
                        have a size of map_dimension **2
        sample_dimension : The size of the feature vector used in clustering
        sample_labels : Labels for the samples that will be used in Training
        """
        self.map_dimension = map_dimension
        self.sample_dimension = sample_dimension
        self.sample_labels = list(set(sample_labels))
        self.W = np.random.normal(loc=0.0,scale=0.01, size=(map_dimension**2,sample_dimension))
        if sample_labels is not None:
            self.Wl = np.zeros((map_dimension**2,len(sample_labels)), dtype=int)


    def _closer_neuron(self, s):
        """
        Returns the index of the closer neuron of the map to the given sample s
        """
        distances = np.array([np.linalg.norm(s-a) for a in self.W])
        min_distances = np.where(distances == distances.min())[0]
        return np.random.choice(min_distances, 1)[0]

    def _distance_in_map(self, n1, n2):
        """
        Returns the distance between two neurons in the map

        n1, n2 : Indexes in the W array
        """
        d1x = n1 / self.map_dimension
        d1y = n1 % self.map_dimension
        d2x = n2 / self.map_dimension
        d2y = n2 % self.map_dimension

        return abs(d1x - d2x) + abs(d1y - d2y)

    def _h(self, d, std):
        """Gauss function"""
        return np.exp(-((d**2) / (2.0*std**2)))

    def train(self, samples, labels=None, learning_rate=1, gauss_std=2):
        """
        Trains the map

        samples : Array of samples used in Training
        labels : Labels for the given samples
        learning_rate : Learning rate to be used in Training
        gauss_std : Standard deviation for the neighbourhood function
        """
        for c, sample in enumerate(samples):
            closer = self._closer_neuron(sample)
            for i, w in enumerate(self.W):
                d = self._distance_in_map(closer, i)
                self.W[i] = w + learning_rate * self._h(d, gauss_std) * (sample - w)

            if (self.sample_labels is not None) and (labels is not None):
                self.Wl[closer, self.sample_labels.index(labels[c])] += 1

    def export_map(self, filename):
        """Exports the map to a .h5 file"""
        out_file = h5py.File(filename, 'w')
        out_file.create_dataset('map', data=self.W)
        out_file.create_dataset('dimension', data=[self.map_dimension])
        out_file.create_dataset('sample_dimension', data=[self.sample_dimension])
        out_file.create_dataset('sample_labels', data=[self.sample_labels])
        if self.sample_labels is not None:
            out_file.create_dataset('map_labels', data=self.Wl)
        out_file.close()

    def import_map(self, filename):
        """Imports the map from a .h5 file"""
        in_file = h5py.File(filename, 'r')
        self.W = np.array([in_file['map']])
        self.map_dimension = in_file['dimension'][0]
        self.sample_dimension = in_file['sample_dimension'][0]
        self.sample_labels = in_file['sample_labels'][0]
        if self.sample_labels is not None:
            self.Wl = np.array(in_file['map_labels'], dtype=int)
        in_file.close()

    def plot_map_labels(self, colors):
        """
        If labels were given in the constructor, it will plot the map colored
        with the color of the dominant label in each neurons.

        colors : List of colors, one for each label
        """
        if self.sample_labels is None:
            print("Map labels were not gathered during training")
            return
        if len(colors) < len(self.sample_labels):
            print("Not enough colors")
            return

        size = 1.0 / self.map_dimension
        patchs = []
        texts = []
        for i, item in enumerate(self.Wl):
            x = i % self.map_dimension
            y = i / self.map_dimension

            win = item.argmax()
            if item[win] == 0:
                alpha = 0
            else:
                #alpha = item[win] / float(sum(item))
                alpha = item[win] / 80.0

            patchs.append(
                patches.Rectangle(
                    (x*size, y*size), size, size,
                    alpha=alpha,
                    facecolor=colors[win])
            )
            texts.append(self.sample_labels[win])

        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        for i, p in enumerate(patchs):
            ax.add_patch(p)
            rx, ry = p.get_xy()
            cx = rx + p.get_width()/2.0
            cy = ry + p.get_height()/2.0
            ax.annotate(texts[i], (cx, cy), color='black', weight='bold',
                fontsize=6, ha='center', va='center')

        plt.show()
