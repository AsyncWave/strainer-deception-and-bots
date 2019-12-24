from flask import Blueprint, jsonify, request, abort, render_template

from flask_cors import cross_origin

from .extensions import mongo 

main = Blueprint('main', __name__)

@main.route('/')
@cross_origin()
def home():
    return render_template('index.html')
    # return jsonify('Server running')

@main.route('/query', methods=['POST'])
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