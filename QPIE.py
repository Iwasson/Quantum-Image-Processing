# Importing standard Qiskit libraries and configuring account
from qiskit import *
from qiskit import IBMQ
from qiskit.compiler import transpile, assemble
from qiskit.tools.jupyter import *
from qiskit.visualization import *
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use('bmh')


import json
from sklearn.metrics import confusion_matrix
from os.path import exists

data = np.load("trainData.npy")
folder = "QPIE_EDGE"


# Initialize some global variable for number of qubits
data_qb = 10
anc_qb = 1
total_qb = data_qb + anc_qb

# Initialize the amplitude permutation unitary
D2n_1 = np.roll(np.identity(2**total_qb), 1, axis=1)

#create the 785 inputs for the perceptron
#scale the data down to be between 0 and 1
#do this by dividing each value by 255
def loadImage(row):
    inputs = []
    for i in range(0,784):
        inputs.append((data[row][i+1])/255)
    return inputs

# Function for plotting the image using matplotlib
def plot_image(img, title: str):
    plt.title(title)
    plt.xticks(range(img.shape[0]))
    plt.yticks(range(img.shape[1]))
    plt.imshow(img, extent=[0, img.shape[0], img.shape[1], 0], cmap='viridis')
    plt.show()
    
# Convert the raw pixel values to probability amplitudes
def amplitude_encode(img_data):
    
    # Calculate the RMS value
    rms = np.sqrt(np.sum(np.sum(img_data**2, axis=1)))
    
    # Create normalized image
    image_norm = []
    for arr in img_data:
        for ele in arr:
            image_norm.append(ele / rms)
        
    # Return the normalized image as a numpy array
    return np.array(image_norm)

def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value


def edgeDetect(imageNum):
    image = np.asarray(loadImage(imageNum)).reshape(28, 28)
    image = np.pad(image, 2, pad_with, padder=0)

    # Get the amplitude ancoded pixel values
    # Horizontal: Original image
    image_norm_h = amplitude_encode(image)

    # Vertical: Transpose of Original image
    image_norm_v = amplitude_encode(image.T)

    # Create the circuit for horizontal scan
    qc_h = QuantumCircuit(total_qb)
    qc_h.initialize(image_norm_h, range(1, total_qb))
    qc_h.h(0)
    qc_h.unitary(D2n_1, range(total_qb))
    qc_h.h(0)
    qc_h.draw('mpl', fold=-1, filename="Circuit_h.png")


    # Create the circuit for vertical scan
    qc_v = QuantumCircuit(total_qb)
    qc_v.initialize(image_norm_v, range(1, total_qb))
    qc_v.h(0)
    qc_v.unitary(D2n_1, range(total_qb))
    qc_v.h(0)
    qc_v.draw('mpl', fold=-1, filename="Circuit_v.png")


    # Combine both circuits into a single list
    circ_list = [qc_h, qc_v]

    # Simulating the cirucits
    back = Aer.get_backend('statevector_simulator')
    results = execute(circ_list, backend=back).result()
    sv_h = results.get_statevector(qc_h)
    sv_v = results.get_statevector(qc_v)

    """
    from qiskit.visualization import array_to_latex
    print('Horizontal scan statevector:')
    array_to_latex(sv_h[:30], max_size=30)
    print()
    print('Vertical scan statevector:')
    array_to_latex(sv_v[:30], max_size=30)
    """
    # Classical postprocessing for plotting the output

    # Defining a lambda function for
    # thresholding to binary values
    threshold = lambda amp: (amp > 1e-15 or amp < -1e-15)

    # Selecting odd states from the raw statevector and
    edge_scan_h = np.abs(np.array([1 if threshold(sv_h[2*i+1].real) else 0 for i in range(2**data_qb)])).reshape(32, 32)
    edge_scan_v = np.abs(np.array([1 if threshold(sv_v[2*i+1].real) else 0 for i in range(2**data_qb)])).reshape(32, 32).T

    # Plotting the Horizontal and vertical scans
    #plot_image(edge_scan_h, 'Horizontal scan output')
    #plot_image(edge_scan_v, 'Vertical scan output')

    # Combining the horizontal and vertical component of the result
    edge_scan_sim = edge_scan_h | edge_scan_v

    np.save(folder + "/" + str(imageNum) + ".npy", edge_scan_sim)

    # Plotting the original and edge-detected images
    #plot_image(image, 'Original image')
    #plot_image(edge_scan_sim, 'Edge Detected image')

for i in range(data.shape[0]):
    edgeDetect(i)
    print("Done with " + str(i))