import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
import joblib
import matplotlib.pyplot as plt
import pandas as pd

# Pre-load fonts (optional)
plt.figure()
plt.close()

mnist= fetch_openml('mnist_784',version=1)

x=mnist['data']
y=mnist=['target'].astype(np.uint8)

rnd_num=x.iloc[0]
rnd_num_img=rnd_num.values.reshape(28,28)
plt.imshow(rnd_num_img,cmap='binary')
plt.axis('off')
plt.show()