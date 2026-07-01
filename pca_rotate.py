import numpy as np
from scipy.signal import savgol_filter
import scipy

def get_eigen(data):
    data = data - np.mean(data, axis=0)
    cov_matrix = np.matmul(data.T,data) / (len(data)-1)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    return eigenvalues, eigenvectors

def pca_rotation(data):
    eigenvalues, eigenvectors = get_eigen(data)
    if np.argmax(eigenvalues) == 1:
        eigenvectors = eigenvectors[:, [1,0]]

    if eigenvectors[0][0] < 0:
        eigenvectors = eigenvectors * np.array([[-1,1]])

    rotated = np.matmul(data, eigenvectors)

    return rotated