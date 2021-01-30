#!/usr/bin/env python3
from flask import Flask, render_template, request, redirect
from core2 import Saati, compute_sentiment, smalltalk, compute_sentiment
from pydantic import BaseModel
import speech_recognition as sr
import uuid
from datetime import datetime

instance = Saati(uuid.uuid4())


class Query(BaseModel):
    uuid: str = uuid.uuid4()
    utterance_ts: datetime = datetime.now()
    input: str
    output: str
    sentiment: str
    score: float


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    transcript = ""
    if request.method == "POST":
        print("FORM DATA RECEIVED")

        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        if file:
            recognizer = sr.Recognizer()
            audioFile = sr.AudioFile(file)
            with audioFile as source:
                data = recognizer.record(source)
            transcript = recognizer.recognize_google(data, key=None)
            print(transcript)
    return render_template("index.html", transcript=transcript)


if __name__ == "__main__":
    app.run(debug=True, threaded=True)
