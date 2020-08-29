import numpy as np

def principal_components(matrix):
    # generate principal components
    covMatrix = np.cov(matrix)
    eigenvalues, components = np.linalg.eig(covMatrix)

    # sort by correspoding eigenvalue
    index = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[index]
    components = components[:, index]
    return eigenvalues, components

matrix = np.random.rand(100,25)

# print(principal_components(matrix))
