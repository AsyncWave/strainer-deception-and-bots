import numpy as np
import pickle

from flask import Flask, request, jsonify, render_template, abort
from flask_pymongo import PyMongo 
from flask_cors import CORS, cross_origin
# MONGO_URI = 'mongodb+srv://strainer_admin:strainer_admin123@strainercluster-igrpg.azure.mongodb.net/strainer?retryWrites=true&w=majority'
mongo = PyMongo()

app = Flask(__name__)

# app.config.from_object('settings')
app.config["MONGO_URI"] = 'mongodb+srv://strainer_admin:strainer_admin123@strainercluster-igrpg.azure.mongodb.net/strainer?retryWrites=true&w=majority'

CORS(app)

app.config['CORS_HEADERS'] = 'Content-Type'

mongo.init_app(app)

model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
@cross_origin()
def home():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
@cross_origin()
def query():
    if not request.json or not 'query' in request.json:
        abort(400)
    query = {
        'name': request.json['screenname'],
        'query': request.json['query'],
        'done': False
    }
    # print(request.json['screenname'])
    query_collection = mongo.db.queries
    query_collection.insert(query)
    return jsonify('Query added successfully'), 201
    # return jsonify({'query': query}), 201

@app.route('/predict',methods=['POST'])
@cross_origin()
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)