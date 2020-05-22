# P5: Eigenfaces
# pca.py
# Name: Oat (Smith) Sukcharoenyingyong
# Net ID: sukcharoenyi@wisc.edu
# CS login: sukcharoenyingyong


import numpy as np
import scipy.linalg
from scipy.io import loadmat
import matplotlib.pyplot
from mpl_toolkits.axes_grid1 import make_axes_locatable


#todo:  load the dataset from a provided .mat file, re-center it around the origin
# and return it as a NumPy array of floats
def load_and_center_dataset(filename):
    dataset = loadmat('YaleB_32x32.mat')
    # get dataset at 'fea'
    x = dataset['fea']
    # center the dataset at the origin
    return x - np.mean(x, axis=0)

#todo: calculate and return the covariance matrix of the dataset as a NumPy matrix (d x d array)
def get_covariance(dataset):
    # covariance matrix of dataset
    return np.dot(np.transpose(dataset), dataset) / (len(dataset) - 1)


#todo:perform eigen decomposition on the covariance matrix S and return a diagonal matrix (NumPy array)
# with the largest m eigenvalues on the diagonal, and a matrix (NumPy array)
# with the corresponding eigenvectors as columns
def get_eig(S, m):
    # get eigenvalues and eigenvectors
    eigVal, eigVec = scipy.linalg.eigh(S)
    # reverse the order of eigenvalues and eigenvectors
    for i in range(len(eigVal) // 2):
        temp = eigVal[i]
        eigVal[i] = eigVal[len(eigVal) - 1 - i]
        eigVal[len(eigVal) - 1 - i] = temp
        for j in range(len(eigVal)):
            temp1 = eigVec[j][i]
            eigVec[j][i] = eigVec[j][len(eigVec) - 1 - i]
            eigVec[j][len(eigVec) - 1 - i] = temp1
    # use slicing to get the first m eigenvalues
    Val = eigVal[:m]
    # use slicing to get the first m eigenvectors
    Vec = eigVec[:, :m]
    # make the eigenvalues an m x m diagonal matrix
    Val = np.diag(Val)
    return Val, Vec


#todo: project each image into your m-dimensional space and return the new representation as a d x 1 NumPy array
def project_image(image, U):
    # create numpy array of length len(U)
    npArr = np.array([0.0] * len(U))
    # for every vector in U multiply the dot product of vector and image to the vector and add
    # it to the numpy array
    for i in range(len(U[0])):
        alpha = np.dot(U[:, i], image)
        npArr += alpha * U[:, i]
    return npArr


#todo: use matplotlib to display a visual representation of the original image and the projected image side-by-side
def display_image(orig, proj):
    # change numpy array of length 1024 to a 2D array of 32 x 32 sides
    orig = orig.reshape(32, 32).transpose()
    proj = proj.reshape(32, 32).transpose()

    # two images in one row
    fig, axs = matplotlib.pyplot.subplots(1, 2)

    # set title for both images
    axs[0].set_title("Original")
    axs[1].set_title("Projection")

    # show the images, create color bar
    color0 = axs[0].imshow(orig, aspect='equal')
    color1 = axs[1].imshow(proj, aspect='equal')

    # make color bar the same height as the images
    divider0 = make_axes_locatable(axs[0])
    cax0 = divider0.append_axes("right", size="5%", pad=0.05)
    divider1 = make_axes_locatable(axs[1])
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)

    # add in the color bar
    fig.colorbar(color0, ax=axs[0], cax=cax0)
    fig.colorbar(color1, ax=axs[1], cax=cax1)

    # display the two images
    matplotlib.pyplot.show()
    return

# x = load_and_center_dataset('YaleB_32x32.mat')
# S = get_covariance(x)
#
# Lambda, U = get_eig(S, 2)
#
# projection = project_image(x[100], U)
# display_image(x[100], projection)

