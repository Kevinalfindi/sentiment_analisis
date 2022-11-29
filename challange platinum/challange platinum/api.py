
from flask import Flask, jsonify
from flask import request
from flasgger import Swagger, LazyString, LazyJSONEncoder
from flasgger import swag_from


import pickle, re
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequence


app = Flask(__name__)
app.json_encoder = LazyJSONEncoder
swagger_template = dict(
    info = {
        'title': LazyString(lambda:'API untuk sentiment analisis'),
        'version': LazyString(lambda:'1.0.0'),
        'description': LazyString(lambda:'Dokumentasi sentiment analisis menggunakan metode NN dan LSTM')
        }, host = LazyString(lambda: request.host)
    )

swagger_config = {
        "headers":[],
        "specs":[
            {
            "endpoint":'docs',
            "route":'/docs.json'
            }
        ],
        "static_url_path":"/flasgger_static",
        "swagger_ui":True,
        "specs_route":"/docs/"
    }

swagger = Swagger(app, template=swagger_template, config=swagger_config)

max_features = 100000
tokenizer = Tokenizer(num_words=max_features, split= ' ', lower=True)

sentiment= ['Negative', 'Positive', 'Neutral']

def cleansing(sent):
    # Mengubah kata menjadi huruf kecil semua dengan menggunakan fungsi lower()
    string = sent.lower()
    # Menghapus emoticon dan tanda baca menggunakan "RegEx" dengan script di bawah
    string = re.sub(r'[^a-zA-Z0-9]', ' ', string)
    return string

def lowercase(string):
    return string.lower() 

def remove_unnecessary_char(string):
    string = re.sub('\n',' ',string) # Remove every '\n'
    string = re.sub('rt',' ',string) # Remove every retweet symbol
    string = re.sub('user',' ',string) # Remove every username
    string = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))',' ',string) # Remove every URL
    string = re.sub('  +', ' ', string) # Remove extra spaces
    string = re.sub(r'pic.twitter.com.[\w]+', '', string) # Remove every pic 
    string = re.sub('gue','saya',string) # replace gue - saya
    string = re.sub(r':', '', string) #Remove symbol 
    string = re.sub(r'‚Ä¶', '', string) #Remove symbol Ä¶
    return string   

file = open('resources_of_LSTM/x_pad_sequences.pickle','rb')
feature_file_from_LSTM = pickle.load(file)
file.close()

model_file_from_LSTM = load.model('model_LSTM/model.h5')

model_file_from_NN = load.model('model_NN/model.p')


@swag_from("docs/LSTM.yml", methods=['POST'])
@app.route('/LSTM', methods=['POST'])
def LSTM():

    original_text = request.form.get('text')
    text = [cleansing(original_text)]
    
    feature = tokenizer.texts_to_sequences(text)
    feature = pad_sequences(feature, maxlen=feature_file_from_LSTM.shape[1])
    
    prediction = model_file_from_LSTM_prediction(feature)
    get_sentiment = sentiment[np.argmax(prediction[0])]

    json_response = {
        'status_code' : 200,
        'description' : "hasil dari sentiment analisis menggunakan LSTM",
        'data' : {
            'text': original_text,
            'sentiment' : get_sentiment
        },
    }
    response_data = jsonify(json_response)
    return response_data

@swag_from("docs/CNN.yml", methods=['POST'])
@app.route('/CNN', methods=['POST'])
def CNN():

    original_text = request.form.get('text')
    text = [cleansing(original_text)]
    
    feature = tokenizer.texts_to_sequences(text)
    
    prediction = model_file_from_NN_prediction(feature)
    get_sentiment = sentiment[np.argmax(prediction[0])]

    json_response = {
        'status_code' : 200,
        'description' : "hasil dari sentiment analisis menggunakan LSTM",
        'data' : {
            'text': original_text,
            'sentiment' : get_sentiment
        },
    }
    response_data = jsonify(json_response)
    return response_data

    if __name__ == '__main__':
        app.run()
