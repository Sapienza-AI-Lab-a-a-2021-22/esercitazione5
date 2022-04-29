import numpy as np

def load_housing_data(features_filename, targets_filename):

    X = np.loadtxt(features_filename)
    y = np.loadtxt(targets_filename)

    return X, y

if __name__ == '__main__':

    file_X = 'datasets/features.dat'
    file_y = 'datasets/targets.dat'

    X, y = load_housing_data(file_X, file_y)

    print(X)
    print(y)
