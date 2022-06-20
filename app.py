import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import h5py    
import numpy as np    
import os

app = Flask(__name__)
model = h5py.File('prediction_model.h5','r')  

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    #output = prediction[0]

    return render_template('index.html', prediction_text='Driving Range should be  $ {}'.format(output))

@app.route('/results', methods=['POST'])
def results():
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__=="__main__":
    app.run(debug=True)
