from flask import Flask, request 
from flask_cors import CORS
import joblib
import numpy as np
from scipy.ndimage import rotate

rnd_clf_model = joblib.load('rnd_clf_model.pkl')

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['POST'])
def predict_react_number():
    if request.method == 'POST': 
        react_number = request.get_json() 
        react_number_np = np.array(react_number) 
        react_number_reshaped = react_number_np.reshape(28, 28)
        react_number_flip = np.flip(react_number_reshaped, 0) 
        react_number_rotated = rotate(react_number_flip, angle=-90) 
        react_number_clean = np.reshape(react_number_rotated, (1,784)) 
        react_number_prediction = rnd_clf_model.predict(react_number_clean) 
        return f"{react_number_prediction[0]}" 
    return 'python-server-v1'
  
if __name__ == '__main__':
    app.run()