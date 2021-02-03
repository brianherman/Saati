#!/usr/bin/env python
# -*- coding: utf-8 -*-
from flask import Flask
from flask import request, sessions
from flask import render_template
import os
import speech_recognition as sr

from core2 import answer_question

app = Flask(__name__)


@app.route("/", methods=["POST", "GET"])
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
    
@app.route('/signUpUser', methods=['POST'])
def signUpUser():
    user =  request.form['username'];
    password = request.form['password'];
    return json.dumps({'status':'OK','user':user,'pass':password});

    
if __name__ == "__main__":
    app.run(debug=True)
