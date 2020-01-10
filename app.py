import numpy as np
import pickle
import hashlib
import json
import re
import requests
import pandas as pd
import string
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()

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
        return jsonify({'message': 'That Tweet is already exists with Id ' + claim + ', please use it to check results','queryId': claim}), 400
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
    result = json.dumps(list(query_collection.find({'queryId' : queryId},{ "_id": 0, "query": 1, "name": 1 })), default=json_util.default)
    if result == "[]":
        return jsonify({'queryId': queryId, 'message': 'No tweet is available for Id ' + id + ', rechek and try again!'}), 400
    # query_collection.update({ "queryId": queryId },{ '$set': { "dataCollected": True }})
    # queryId = (str(re.findall("\d+", result))).replace('[','').replace(']','').replace('\'','')
    # return jsonify({'message': 'Query exists', 'queryId': queryId, result}), 200
    return result, 200

@app.route('/credibility', methods=['POST'])
@cross_origin()
def credibility():
    if not request.json or not 'tweet' in request.json:
        abort(400)
    tweet = request.json['tweet']
    cvec = pickle.load(open("vectorizer.pickle", 'rb'))
    model = pickle.load(open("model_credibility.pickle", 'rb'))
    msg = cvec.transform([tweet])
    pred = model.predict(msg)
    prediction = int(pred[0])
    return jsonify({'prediction': prediction}), 200

@app.route('/bot/<user>', methods=['GET'])
@cross_origin()
def bot(user):
    r = requests.get('https://strainer-twitter-server.herokuapp.com/exists/'+user)
    responce = r.json()
    try:
        screen_name = responce[0]['screen_name']
    except:
        return jsonify({'message': 'Not a valid screen name'}), 404

    try:
        location = responce[0]['location']
        description = responce[0]['description']
        url = responce[0]['url']
        created_at = responce[0]['created_at']
        lang = "" if responce[0]['lang'] == None else responce[0]['lang']
        status = responce[0]['status']
        has_extended_profile = responce[0]['has_extended_profile']
        name = responce[0]['name']
        verified = responce[0]['verified']
        followers_count = responce[0]['followers_count']
        friends_count = responce[0]['friends_count']
        statuses_count = responce[0]['statuses_count']
        listed_count = responce[0]['listed_count']
    except:
        return jsonify({'message': 'Somrthing went wrong'}), 400
        
    
    df_bot_test = pd.DataFrame(columns=["screen_name", "location", "description","url","created_at","lang","status","has_extended_profile","name","verified","followers_count","friends_count","statuses_count","listed_count"], data=[[screen_name, location, description, url, created_at,lang,status,has_extended_profile,name,verified,followers_count,friends_count, statuses_count,listed_count]])
    text_cols = df_bot_test[['screen_name','location','description','url','created_at','lang','status','has_extended_profile','name']].copy()
    text_cols.rename(columns={'screen_name':'screen_name_processed'}, inplace=True)
    text_cols.rename(columns={'name':'name_processed'}, inplace=True)

    variable = 'screen_name'
    stop_words = set(stopwords.words('english'))
    text_cols[variable+'_processed_num_count'] = ""
    for i, row1 in text_cols.iterrows():
        row1[variable+'_processed'] = row1[variable+'_processed'].lower() #Convert text to lowercase
        row1[variable+'_processed_num_count'] = sum(ch.isdigit() for ch in row1[variable+'_processed']) #create new column to get number of numbers
        row1[variable+'_processed'] = re.sub(r'\d+','', row1[variable+'_processed']) #Remove numbers
        row1[variable+'_processed']= row1[variable+'_processed'].translate(str.maketrans('','',string.punctuation)) #Remove punctuation
        row1[variable+'_processed'] = row1[variable+'_processed'].strip() #Remove whitespaces
        row1[variable+'_processed'] = [i for i in word_tokenize(row1[variable+'_processed']) if not i in stop_words] #Tokenization - REMOVE STOP WORDS
        for word in row1[variable+'_processed']:
            row1[variable+'_processed'] = lemmatizer.lemmatize(word)
        text_cols.at[i, 'screen_name_processed'] = row1['screen_name_processed']
        text_cols.at[i, 'screen_name_processed_num_count'] = row1['screen_name_processed_num_count']

    text_cols['name_processed_num_count'] = ""
    for index, row in text_cols.iterrows():
        row['name_processed'] = row['name_processed'].lower() #Convert text to lowercase
        row['name_processed_num_count'] = (sum(c.isdigit() for c in row['name_processed'])) #create new column to get number of numbers
        row['name_processed'] = re.sub(r'\d+','', row['name_processed']) #Remove numbers
        row['name_processed']= row['name_processed'].translate(str.maketrans('','',string.punctuation)) #Remove punctuation
        row['name_processed'] = row['name_processed'].strip() #Remove whitespaces
        row['name_processed'] = [i for i in word_tokenize(row['name_processed']) if not i in stop_words] #Tokenization - REMOVE STOP WORDS
        text_cols.at[index, 'name_processed'] = row['name_processed']
        text_cols.at[index, 'name_processed_num_count'] = row['name_processed_num_count']

    word_list = r'bot|b0t|cannabis|tweet me|mishear|follow me|updates every|gorilla|yes_ofc|forget' \
                    r'expos|kill|clit|bbb|butt|fuck|XXX|sex|truthe|fake|anony|free|virus|funky|RNA|kuck|jargon' \
                    r'nerd|swag|jack|bang|bonsai|chick|prison|paper|pokem|xx|freak|ffd|dunia|clone|genie|bbb' \
                    r'ffd|onlyman|emoji|joke|troll|droop|free|every|wow|cheese|yeah|bio|magic|wizard|face'
    listofwords = pickle.load(open("words_in_not_credible.pickle", 'rb'))
    listofwords2 = pickle.load(open("word_couples_in_not_credible.pickle", 'rb'))
    list_ofwords=list(listofwords)
    str1 = '|'.join(str(e) for e in list_ofwords)
    list_ofwords2=list(listofwords2)
    str2 = '|'.join(str(e) for e in list_ofwords2)
    word_list = str1 + '|' + str2 + '|' + word_list

    text_cols['screen_name_binary'] = df_bot_test.screen_name.str.contains(word_list, case=False, na=False)
    text_cols['name_binary'] = df_bot_test.name.str.contains(word_list, case=False, na=False)
    text_cols['description_binary'] = df_bot_test.description.str.contains(word_list, case=False, na=False)
    text_cols['status_binary'] = df_bot_test.status.str.contains(word_list, case=False, na=False)
    text_cols['listed_count_binary'] = (df_bot_test.listed_count>20000)==False

    for column in df_bot_test:
        text_cols[column+'_NA'] = np.where(df_bot_test[column].isnull(), 1, 0)
    
    variable = 'has_extended_profile'
    for i, row in text_cols[text_cols[variable].isnull()].iterrows():
        obs_sample = text_cols[variable].dropna().sample(1, random_state=int(row.screen_name_processed_num_count))
        obs_sample.index = [i]
        text_cols.at[i, variable] = obs_sample

    text_cols.has_extended_profile = text_cols.has_extended_profile.astype(int)

    text_cols.rename(columns={'has_extended_profile':'has_extended_profile_processed'}, inplace=True)
    text_cols_features = text_cols[['has_extended_profile_processed','name_processed_num_count','screen_name_processed_num_count','screen_name_binary', 'name_binary', 'description_binary', 'status_binary', 'listed_count_binary','location_NA','description_NA','url_NA','status_NA','has_extended_profile_NA']].copy()
    test_data_features = df_bot_test[['verified', 'followers_count', 'friends_count', 'statuses_count']].copy()

    result = pd.concat([text_cols_features, test_data_features], axis=1, sort=False)
    point = result.head(1).to_numpy()

    model_bot = pickle.load(open("model_bot.pickle", 'rb'))
    pred = model_bot.predict(point.reshape(1, -1))
    prediction = int(pred[0])
    # cvec = pickle.load(open("vectorizer.pickle", 'rb'))
    # model = pickle.load(open("model_credibility.pickle", 'rb'))
    # msg = cvec.transform([tweet])
    # pred = model.predict(msg)
    # predictoin = int(pred[0])
    return jsonify({'prediction': prediction}), 200

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