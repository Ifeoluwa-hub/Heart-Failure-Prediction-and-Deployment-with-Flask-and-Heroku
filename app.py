import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    final_features = scaler.transform(final_features)    
    prediction = model.predict(final_features)
    print("final features",final_features)
    print("prediction:",prediction)
    output = round(prediction[0], 2)
    print(output)

    if output == 0:
        return render_template('index.html', prediction_text='THE PATIENT IS NOT LIKELY TO HAVE A HEART FAILURE')
    else:
         return render_template('index.html', prediction_text='THE PATIENT IS LIKELY TO HAVE A HEART FAILURE')
        
@app.route('/predict_api',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=False)