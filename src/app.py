#!/usr/bin/env python
# -*- coding: utf-8 -*-
from flask import Flask
from flask import request, sessions, jsonify
from flask import render_template
#from config import *
import os
import speech_recognition as sr
from translate import Translator
from core2 import answer_question

app = Flask(__name__)
#app.config["DEBUG"] = True # turn off in prod
translator = Translator(MODEL_PATH)

@app.route("/input", methods=["POST", "GET"])
def index():
    if request.method == "POST":

        file = request.files["audio_data"]
        # with open('audio.wav', 'wb') as audio:
        #    f.save(audio)
        recognizer = sr.Recognizer()
        audioFile = sr.AudioFile(file)
        with audioFile as source:
            data = recognizer.record(source)
        transcript = recognizer.recognize_google(data, key=None)
        print(transcript)
        response = answer_question(transcript)[0]
        transcript = transcript + response
        print("file uploaded successfully")
        return render_template("index2.html", request="POST")
    else:
        return render_template("index2.html")

@app.route('/', methods=["GET"])
def health_check():
    """Confirms service is running"""
    return "Machine translation service is up and running."

@app.route('/lang_routes', methods = ["GET"])
def get_lang_route():
    lang = request.args['lang']
    all_langs = translator.get_supported_langs()
    lang_routes = [l for l in all_langs if l[0] == lang]
    return jsonify({"output":lang_routes})

@app.route('/supported_languages', methods=["GET"])
def get_supported_languages():
    langs = translator.get_supported_langs()
    return jsonify({"output":langs})

@app.route('/translate', methods=["POST"])
def get_prediction():
    source = request.json['source']
    target = request.json['target']
    text = request.json['text']
    translation = translator.translate(source, target, text)
    return jsonify({"output":translation})
    
@app.route('/signUpUser', methods=['POST'])
def signUpUser():
    user =  request.form['username'];
    password = request.form['password'];
    return json.dumps({'status':'OK','user':user,'pass':password});
    
if __name__ == "__main__":
    app.run(debug=True)


