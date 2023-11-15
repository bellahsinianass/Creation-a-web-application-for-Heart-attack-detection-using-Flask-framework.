from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__,template_folder='template')

# Load the pickled model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    return render_template('index.html', prediction_text='Death Event Prediction: {}'.format(prediction[0]))

if __name__ == "__main__":
    app.run(debug=True)
