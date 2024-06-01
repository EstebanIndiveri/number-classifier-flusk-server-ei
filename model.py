import numpy as np

from sklearn.datasets improt fetch_openml
from  sklearn.ensemble import RandomForestClassifier

from joblib
import matplotlib.pyplot as pyplot
import pandas as pd 

mnist= fetch_openml('mnist_784',version=1)

x=mnist['data']
y=mnist=['target'].astype(np.uint8)

rnd_num=x.iloc[0];