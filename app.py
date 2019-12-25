import numpy as np
import pickle
import hashlib
import json
import re

from bson import json_util
from flask import Flask, request, jsonify, render_template, abort
from flask_pymongo import PyMongo 
from flask_cors import CORS, cross_origin

mongo = PyMongo()

app = Flask(__name__)

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
    queryId = int(hashlib.sha256(request.json['query'].encode('utf-8')).hexdigest(), 16) % 10**8
    query = {
        'queryId': queryId,
        'name': request.json['screenname'],
        'query': request.json['query'],
        'dataCollected': False,
        'credibility': False,
        'profile': False,
        'network': False
    }
    query_collection = mongo.db.queries
    result = json.dumps(list(query_collection.find({'queryId' : queryId},{ "_id": 0, "queryId": 1 })), default=json_util.default)
    if result != "[]":
        claim = (str(re.findall("\d+", result))).replace('[','').replace(']','').replace('\'','')
        # claim = re.sub('[[]]', '', claim)
        return jsonify({'message': 'That Tweet is already exists with Id ' + claim + ', please use it to check results'}), 400
    query_collection.insert(query)
    return jsonify({'queryId': queryId,'message': 'Tweet added successfully'}), 201

@app.route('/nodata', methods=['GET'])
@cross_origin()
def nodata():
    query_collection = mongo.db.queries
    docs_list = list(query_collection.find({'dataCollected' : False},{ "_id": 0, "queryId": 1, "query": 1 }))
    return json.dumps(docs_list, default=json_util.default, indent=4, sort_keys=True)

@app.route('/setdata/<id>', methods=['POST'])
@cross_origin()
def setdata(id):
    try:
        queryId = int(id)
    except:
        return jsonify({'message': 'Not a valid Id'}), 400
    
    query_collection = mongo.db.queries
    result = json.dumps(list(query_collection.find({'queryId' : queryId},{ "_id": 0, "queryId": 1 })), default=json_util.default)
    if result == "[]":
        return jsonify({'queryId': queryId, 'message': 'No tweet is available for that Id'}), 400
    query_collection.update({ "queryId": queryId },{ '$set': { "dataCollected": True }})
    return jsonify({'message': 'Tweet updated as data collected'}), 200

@app.route('/get/<id>', methods=['GET'])
@cross_origin()
def get(id):
    try:
        queryId = int(id)
    except:
        return jsonify({'message': 'Not a valid Id'}), 400
    
    query_collection = mongo.db.queries
    result = json.dumps(list(query_collection.find({'queryId' : queryId},{ "_id": 0, "query": 1 })), default=json_util.default)
    if result == "[]":
        return jsonify({'queryId': queryId, 'message': 'No tweet is available for Id ' + id + ', rechek and try again!'}), 400
    # query_collection.update({ "queryId": queryId },{ '$set': { "dataCollected": True }})
    # queryId = (str(re.findall("\d+", result))).replace('[','').replace(']','').replace('\'','')
    # return jsonify({'message': 'Query exists', 'queryId': queryId, result}), 200
    return result, 200

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