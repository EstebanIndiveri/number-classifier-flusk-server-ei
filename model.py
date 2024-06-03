import numpy as np
import ssl

ssl._create_default_https_context = ssl._create_stdlib_context
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
y=mnist['target'].astype(np.uint8)
# y=mnist=['target'].astype(data = np.array(data).astype('float32'))
print(mnist)
x_all_black=x.replace([range(1,255)],255)

rnd_clf=RandomForestClassifier(n_estimators=500,max_leaf_nodes=16,n_jobs=-1)
rnd_clf.fit(x_all_black,y)

joblib.dump(rnd_clf,'rnd_clf_model.pkl')
print('entrenamiento terminado')

### SET random numer without forest 
# rnd_num=x_all_black.iloc[8]
# rnd_num_img=rnd_num.values.reshape(28,28)
# plt.imshow(rnd_num_img,cmap='binary')
# plt.axis('off')
# plt.show()

