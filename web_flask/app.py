import os
import sys
import hashlib

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect, send_from_directory
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Some utilites
import numpy as np

from inference.gezixi.gezixi_inference import GeZiXiInfer
from utils.hparams import set_hparams, hparams
from utils.audio import save_wav
import random
import traceback
from zhon.hanzi import punctuation as zhp
import string

# Declare a flask app
app = Flask(__name__)
app.config['RESULTS_FOLDER'] = 'static/results/'

# set hparams
set_hparams(config='usr/configs/gezixi.yaml', exp_name='gezixi_fs2midi_per_fm_adv_m_discs_hok_pre_data_aug', print_hparams=False)

# load model
model = GeZiXiInfer(hparams)
print('Model loaded.')

def extends_lyrics(input_dict):
    punc = string.punctuation + zhp
    lyrics = [i for i in input_dict['text'] if i != ' ']
    lyrics = [i if i not in punc else 'sp' for i in lyrics]
    if lyrics[0] != 'sp':
        lyrics.insert(0, 'sp')
    notes = [x.strip() for x in input_dict['notes'].split(' ') if x.strip() != '']
    while len(lyrics) < len(notes):
        idx = random.randint(2, len(lyrics) - 1)
        if lyrics[idx - 1] == 'sp':
            continue
        lyrics.insert(idx, '-')
    for idx, token in enumerate(lyrics):
        if token == 'sp':
            notes[idx] = 'sp'
    input_dict['text'] = ' '.join(lyrics)
    input_dict['notes'] = ' '.join(notes)

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('base.html')

@app.route('/p', methods=['GET'])
def p_index():
    # Main page
    return render_template('professional.html')

@app.route('/a', methods=['GET'])
def a_index():
    # Main page
    return render_template('amateur.html')


@app.route('/professional', methods=['GET', 'POST'])
def p_predict():
    if request.method == 'POST':
        # Make prediction
        try:
            result = model.infer_once(request.json)
            print(f"Sung by {request.json['spk_name']}")
            item_label = request.json['spk_name'] + request.json['text'] + request.json['notes'] + request.json['notes_duration']
            item_name = str(hashlib.md5(item_label.encode("utf8")).hexdigest())
            wav_name = f'web_flask/static/results/{item_name}.wav'

            # Serialize the result, you can add additional fields
            save_wav(result, wav_name, hparams['audio_sample_rate'])

            return {'audioUrl': f'results/{item_name}.wav', 'status': True}
        except:
            traceback.print_exc()
            return {'audioUrl': None, 'status': False}

    return None

@app.route('/amateur', methods=['GET', 'POST'])
def a_predict():
    if request.method == 'POST':
        # Make prediction
        try:
            input_dict = request.json
            extends_lyrics(input_dict)
            result = model.infer_once(input_dict)
            print(f"Sung by {request.json['spk_name']}")
            item_label = request.json['spk_name'] + request.json['text'] + request.json['notes'] + request.json['notes_duration']
            item_name = str(hashlib.md5(item_label.encode("utf8")).hexdigest())
            wav_name = f'web_flask/static/results/{item_name}.wav'

            # Serialize the result, you can add additional fields
            save_wav(result, wav_name, hparams['audio_sample_rate'])

            return {'audioUrl': f'results/{item_name}.wav', 'status': True}
        except:
            traceback.print_exc()
            return {'audioUrl': None, 'status': False}

    return None

@app.route('/results/<filename>', methods=['GET'])
def download(filename):
    if request.method == "GET":
        path = os.path.isfile(os.path.join('web_flask', app.config['RESULTS_FOLDER'], filename));
        
        if path:
            return send_from_directory(app.config['RESULTS_FOLDER'], filename, as_attachment=True)
        else:
            print(f"{os.path.join('web_flask', app.config['RESULTS_FOLDER'], filename)} not found!")
    
    return None


if __name__ == '__main__':
    # app.run(port=5002, threaded=False)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5002), app)
    http_server.serve_forever()
