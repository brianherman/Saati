#!/usr/bin/env python3
import gradio as gr
from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel

import csv, smtplib, uuid, logging, os, pickle, json
from transitions.extensions import HierarchicalGraphMachine as Machine
from core2 import Saati, compute_sentiment, smalltalk, compute_sentiment

instance = Saati(uuid.uuid4())


class Event(BaseModel):
    uuid: str = uuid.uuid4()
    utterance_ts: datetime = datetime.now()
    input: str
    output: List[str]
    sentiment: int
    sync_ratio: float
    interactions: int


def greetMe():
    CurrentHour = int(datetime.now().hour)
    if CurrentHour >= 0 and CurrentHour < 12:
        talk("Good Morning!")

    elif CurrentHour >= 12 and CurrentHour < 18:
        talk("Good Afternoon!")

    elif CurrentHour >= 18 and CurrentHour != 0:
        talk("Good Evening!")


def greet(name):
    return "Hello " + name + "!"


def journal_sleep(response: str):
    CurrentHour = int(datetime.now().hour)
    if CurrentHour >= 0 and CurrentHour < 9:
        talk(" How well did you sleep ? ")
    elif CurrentHour >= 10 and CurrentHour <= 12:
        talk(" Did you sleep in? ")
    return response

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}


@app.post("/events/")
async def create_item(item: Item):
    return item


import tempfile

app = Flask(__name__)

@app.post('/process_utterance', methods=['POST'])
def pitch_track():
    import parselmouth

    # Save the file that was sent, and read it into a parselmouth.Sound
    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(request.files['audio'].read())
        sound = parselmouth.Sound(tmp.name)

    # Calculate the pitch track with Parselmouth
    #pitch_track = sound.to_pitch().selected_array['frequency']

    # Convert the NumPy array into a list, then encode as JSON to send back
    return jsonify(list(pitch_track))
