import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__) #Initialize the flask App
model = pickle.load(open('model.pkl', 'rb')) # loading the trained model

@app.route('/') # Homepage
def home():
    return render_template('main.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    
    # retrieving values from form
    init_features = [float(x) for x in request.form.values()]
    final_features = [np.array(init_features)]

    prediction = model.predict(final_features) # making prediction

    output = prediction[0]
    if (output > 100):
        output = 100
    if (init_features[0] > 24):
        return render_template('main.html', prediction_text="A Day can't have more than 24 hrs")
    else:
        return render_template('main.html',
                               prediction_text='Expected percentage for {} hours study is: {:.2f}%'.format(init_features[0],output))


if __name__ == "__main__":
    app.run(debug=True)