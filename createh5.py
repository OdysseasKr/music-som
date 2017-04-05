from __future__ import print_function
import librosa
import numpy as np
import h5py
import sys
import os

def create_dataset(input_path, output_path):
    print("Creating h5 from {} to file {}".format(input_path, output_path))

    feature_vector_dim = 35
    labels = os.listdir(input_path)
    data_matrix = np.empty((len(labels)*100, feature_vector_dim))
    data_labels = np.chararray((len(labels)*100,1), itemsize=10)
    index = 0
    for l, label in enumerate(labels):
        print("Data for {}".format(label))

        instrument_dir = os.path.join(input_path, label)
        files = os.listdir(instrument_dir)
        skipped = 0

        # Read files for each genre
        for i, track in enumerate(files):
            print(" {} of {}".format(i+1,len(files)), end="\r")
            sys.stdout.flush()

            try:
                y, sr = librosa.load(os.path.join(instrument_dir, track))
                stft = np.abs(librosa.stft(y))

                arr = np.empty(feature_vector_dim)
                arr[0] = librosa.beat.beat_track(y, sr)[0]
                arr[1] = librosa.estimate_tuning(y, sr)
                arr[2:8] = np.mean(librosa.feature.tonnetz(librosa.effects.harmonic(y), sr), axis=1)
                arr[8:28] = np.mean(librosa.feature.mfcc(y, sr), axis=1)
                arr[28:] = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr), axis=1)

                data_matrix[index] = arr
                data_labels[index] = label
                index += 1
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                skipped += 1
        print("")


        if skipped > 0:
            print(" Skipped {} corrupted files".format(skipped))


    # Resize array
    data_matrix = data_matrix[:index]
    data_labels = data_labels[:index]

    # Normalise
    minis = np.min(data_matrix, axis=0)
    data_matrix2 = data_matrix + minis
    minis = np.min(data_matrix2, axis=0)
    maxis = np.max(data_matrix2, axis=0)
    diff = maxis - minis
    diff[diff==0] = 1
    data_matrix = (data_matrix2 - minis) / diff

    # Write to file
    out_file = h5py.File(output_path, 'w')
    for l, label in enumerate(labels):
        out_file.create_dataset(label, data=data_matrix[(data_labels == label).flatten()])
    out_file.create_dataset('vector_size', data=[feature_vector_dim])
    out_file.close()

    print("Done")

if __name__ == '__main__':
    if (len(sys.argv) < 3):
        print("Usage: python createh5 input_path output_path")
        exit()

    create_dataset(sys.argv[1], sys.argv[2])
