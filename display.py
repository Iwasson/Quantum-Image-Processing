import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use('bmh')

folder = "QPIE_EDGE/"
data = np.load(folder + "4.npy")

# Function for plotting the image using matplotlib
def plot_image(img, title: str):
    plt.title(title)
    plt.xticks(range(img.shape[0]))
    plt.yticks(range(img.shape[1]))
    plt.imshow(img, extent=[0, img.shape[0], img.shape[1], 0], cmap='viridis')
    plt.show()

#plot_image(data, 'test')

test = np.load("trainData.npy")
print(test.shape[0])