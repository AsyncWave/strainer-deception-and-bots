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
nltk.download('punkt')
nltk.download('wordnet')
import traceback

from werkzeug.wsgi import ClosingIterator
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

class AfterThisResponse:
    def __init__(self, app=None):
        self.callbacks = []
        if app:
            self.init_app(app)

    def __call__(self, callback):
        self.callbacks.append(callback)
        return callback

    def init_app(self, app):
        # install extensioe
        app.after_this_response = self

        # install middleware
        app.wsgi_app = AfterThisResponseMiddleware(app.wsgi_app, self)

    def flush(self):
        try:
            for fn in self.callbacks:
                try:
                    fn()
                except Exception:
                    traceback.print_exc()
        finally:
            self.callbacks = []

class AfterThisResponseMiddleware:
    def __init__(self, application, after_this_response_ext):
        self.application = application
        self.after_this_response_ext = after_this_response_ext

    def __call__(self, environ, start_response):
        iterator = self.application(environ, start_response)
        try:
            return ClosingIterator(iterator, [self.after_this_response_ext.flush])
        except Exception:
            traceback.print_exc()
            return iterator

AfterThisResponse(app)

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
        'keyword_list': request.json['keyword_list'],
        'botAmount': 0,
        'credAmount': 0,
        'dataCollected': False,
        'credibility': False,
        'profile': False,
        'network': False,
        'forecast': 0
    }

    queryToSend={}
    queryToSend['queryId'] = queryId
    queryToSend['query']= request.json['query']
    queryToSend['keyword_list']= request.json['keyword_list']
    print("request.json['keyword_list'] >>>>",request.json['keyword_list'])
    print("queryToSend['keyword_list'] >>>>",queryToSend['keyword_list'])

    query_collection = mongo.db.queries
    result = json.dumps(list(query_collection.find({'queryId' : queryId},{ "_id": 0, "queryId": 1 })), default=json_util.default)
    if result != "[]":
        claim = (str(re.findall("\d+", result))).replace('[','').replace(']','').replace('\'','')
        # claim = re.sub('[[]]', '', claim)
        return jsonify({'message': 'That Tweet is already exists with Id ' + claim + ', please use it to check results','queryId': claim}), 400
    query_collection.insert(query)

    response = requests.post('https://strainer-data-demo.herokuapp.com/query', json=queryToSend)
    if response.status_code != 201:
        abort(400)
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

    @app.after_this_response
    def post_process():
        responce = requests.get('https://strainer-data-demo.herokuapp.com/get/'+ str(queryId))
        cvec = pickle.load(open("vectorizer.pickle", 'rb'))
        model = pickle.load(open("model_credibility.pickle", 'rb'))
        model_bot = pickle.load(open("model_bot.pickle", 'rb'))

        total = 0
        total_bot = 0
        cred = 0
        bot = 0
        for tweet in responce.json(): 
            # print(tweet[0])
            total = total + 1
            tweet0 = tweet['tweet']
            msg = cvec.transform([tweet0])
            pred0 = model.predict(msg)
            prediction0 = int(pred0[0])
            print(total)
            if (prediction0 == 1):
                cred = cred + 1

        totalcred = (cred/total)*100
        query_collection.update({ "queryId": queryId },{ '$set': { "credAmount": totalcred }})

        for tweet in responce.json():
            try:
                location = tweet['location']
                description = tweet['description']
                url = tweet['url']
                created_at = tweet['created_at']
                lang = "" if tweet['lang'] == None else tweet['lang']
                status = tweet['tweet']
                has_extended_profile = tweet['has_extended_profile']
                name = tweet['name']
                verified = tweet['verified']
                followers_count = tweet['followers_count']
                friends_count = tweet['friends_count']
                statuses_count = tweet['statuses_count']
                listed_count = tweet['listed_count']
                screen_name = tweet['screen_name']
                total_bot = total_bot + 1
                print(total_bot)
            except:
                print('error on bot detection')
                # return jsonify({'message': 'Somrthing went wrong'}), 400
                
            
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

            df_bot_test['des_hashtags'] = df_bot_test['description'].str.count('#')
            df_bot_test['des_mentions'] = df_bot_test['description'].str.count('@')
            df_bot_test['des_length'] = df_bot_test['description'].str.len()
            df_bot_test['status_hashtags'] = df_bot_test['status'].str.count('#')
            df_bot_test['status_mentions'] = df_bot_test['status'].str.count('@')
            df_bot_test['status_length'] = df_bot_test['status'].str.len()
            df_bot_test['des_link_count'] = df_bot_test['description'].str.count(':')

            df_bot_test['status_punctuation'] = df_bot_test['status'].str.count('\.')
            df_bot_test['des_punctuation'] = df_bot_test['description'].str.count('\.')
            df_bot_test['status_quote'] = df_bot_test['status'].str.count('"')
            df_bot_test['des_quote'] = df_bot_test['description'].str.count('"')

            feature_set = df_bot_test[['status_punctuation','des_punctuation','status_quote','des_quote','des_link_count','des_hashtags', 'des_mentions', 'des_length', 'status_hashtags','status_mentions','status_length']].copy().fillna(0)
            # feature_set = df_bot_test[['des_link_count', 'des_hashtags', 'des_mentions', 'des_length', 'status_hashtags','status_mentions','status_length']].copy().fillna(0)

            text_cols.rename(columns={'has_extended_profile':'has_extended_profile_processed'}, inplace=True)
            text_cols_features = text_cols[['has_extended_profile_processed','name_processed_num_count','screen_name_processed_num_count','screen_name_binary', 'name_binary', 'description_binary', 'status_binary', 'listed_count_binary','location_NA','description_NA','url_NA','status_NA','has_extended_profile_NA']].copy()
            test_data_features = df_bot_test[['verified', 'followers_count', 'friends_count', 'statuses_count']].copy()

            result = pd.concat([feature_set, text_cols_features, test_data_features], axis=1, sort=False)
            point = result.head(1).to_numpy()

            pred1 = model_bot.predict(point.reshape(1, -1))
            prediction1 = int(pred1[0])
            if (prediction1 == 1):
                bot = bot + 1
        
        totalbot = (bot/total_bot)*100
        query_collection.update({ "queryId": queryId },{ '$set': { "botAmount": totalbot }})

        query_collection.update({ "queryId": queryId },{ '$set': { "credibility": True }})

    return jsonify({'message': 'Tweet updated'}), 200

@app.route('/final/<dashboard>/<id>', methods=['GET'])
@cross_origin()
def final(dashboard,id):
    try:
        queryId = int(id)
        selector = int(dashboard)
    except:
        return jsonify({'message': 'Not a valid Id or Selector'}), 400

    query_collection = mongo.db.queries
    if (dashboard == '0'):
        result = json.dumps(list(query_collection.find({'queryId' : queryId},{ "_id": 0, "credAmount": 1 })), default=json_util.default)
        if result == "[]":
            return jsonify({'queryId': queryId, 'message': 'No tweet is available for that Id'}), 404
        return result, 200

    elif (dashboard == '1'):
        result = json.dumps(list(query_collection.find({'queryId' : queryId},{ "_id": 0, "botAmount": 1 })), default=json_util.default)
        if result == "[]":
            return jsonify({'queryId': queryId, 'message': 'No tweet is available for that Id'}), 404
        return result, 200
    else:
        abort(400)

@app.route('/setcredibility/<id>', methods=['POST'])
@cross_origin()
def setcredibility(id):
    try:
        queryId = int(id)
    except:
        return jsonify({'message': 'Not a valid Id'}), 400
    
    query_collection = mongo.db.queries
    result = json.dumps(list(query_collection.find({'queryId' : queryId},{ "_id": 0, "queryId": 1 })), default=json_util.default)
    if result == "[]":
        return jsonify({'queryId': queryId, 'message': 'No tweet is available for that Id'}), 400
    query_collection.update({ "queryId": queryId },{ '$set': { "credibility": True }})
    return jsonify({'message': 'Tweet updated as credibility calculated'}), 200

@app.route('/setprofile/<id>', methods=['POST'])
@cross_origin()
def setprofile(id):
    try:
        queryId = int(id)
    except:
        return jsonify({'message': 'Not a valid Id'}), 400
    
    query_collection = mongo.db.queries
    result = json.dumps(list(query_collection.find({'queryId' : queryId},{ "_id": 0, "queryId": 1 })), default=json_util.default)
    if result == "[]":
        return jsonify({'queryId': queryId, 'message': 'No tweet is available for that Id'}), 400
    query_collection.update({ "queryId": queryId },{ '$set': { "profile": True }})
    return jsonify({'message': 'Tweet updated as profile calculated'}), 200

@app.route('/setnetwork/<id>', methods=['POST'])
@cross_origin()
def setnetwork(id):
    try:
        queryId = int(id)
    except:
        return jsonify({'message': 'Not a valid Id'}), 400
    
    query_collection = mongo.db.queries
    result = json.dumps(list(query_collection.find({'queryId' : queryId},{ "_id": 0, "queryId": 1 })), default=json_util.default)
    if result == "[]":
        return jsonify({'queryId': queryId, 'message': 'No tweet is available for that Id'}), 400
    query_collection.update({ "queryId": queryId },{ '$set': { "network": True }})
    return jsonify({'message': 'Tweet updated as network calculated'}), 200

@app.route('/progress/<id>', methods=['GET'])
@cross_origin()
def progress(id):
    try:
        queryId = int(id)
    except:
        return jsonify({'message': 'Not a valid Id'}), 400
    
    query_collection = mongo.db.queries
    result = json.dumps(list(query_collection.find({'queryId' : queryId},{ "_id": 0, "queryId": 0, "name": 0, "query": 0})), default=json_util.default)
    if result == "[]":
        return jsonify({'queryId': queryId, 'message': 'No tweet is available for that Id'}), 400
    return result, 200

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

#region Credibbility
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
#endregion

#region Automated Account Detection
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

    df_bot_test['des_hashtags'] = df_bot_test['description'].str.count('#')
    df_bot_test['des_mentions'] = df_bot_test['description'].str.count('@')
    df_bot_test['des_length'] = df_bot_test['description'].str.len()
    df_bot_test['status_hashtags'] = df_bot_test['status'].str.count('#')
    df_bot_test['status_mentions'] = df_bot_test['status'].str.count('@')
    df_bot_test['status_length'] = df_bot_test['status'].str.len()
    df_bot_test['des_link_count'] = df_bot_test['description'].str.count(':')

    df_bot_test['status_punctuation'] = df_bot_test['status'].str.count('\.')
    df_bot_test['des_punctuation'] = df_bot_test['description'].str.count('\.')
    df_bot_test['status_quote'] = df_bot_test['status'].str.count('"')
    df_bot_test['des_quote'] = df_bot_test['description'].str.count('"')

    feature_set = df_bot_test[['status_punctuation','des_punctuation','status_quote','des_quote','des_link_count','des_hashtags', 'des_mentions', 'des_length', 'status_hashtags','status_mentions','status_length']].copy().fillna(0)
    # feature_set = df_bot_test[['des_link_count', 'des_hashtags', 'des_mentions', 'des_length', 'status_hashtags','status_mentions','status_length']].copy().fillna(0)

    text_cols.rename(columns={'has_extended_profile':'has_extended_profile_processed'}, inplace=True)
    text_cols_features = text_cols[['has_extended_profile_processed','name_processed_num_count','screen_name_processed_num_count','screen_name_binary', 'name_binary', 'description_binary', 'status_binary', 'listed_count_binary','location_NA','description_NA','url_NA','status_NA','has_extended_profile_NA']].copy()
    test_data_features = df_bot_test[['verified', 'followers_count', 'friends_count', 'statuses_count']].copy()

    result = pd.concat([feature_set, text_cols_features, test_data_features], axis=1, sort=False)
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
#endregion

if __name__ == "__main__":
    app.run(debug=True)