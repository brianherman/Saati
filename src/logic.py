#!/usr/bin/env python3
import torch
import numpy as np
from scipy.io.wavfile import write
from transformers import (
    TFAutoModelWithLMHead,
    AutoTokenizer,
    pipeline,
    BlenderbotTokenizer,
    BlenderbotForConditionalGeneration,
    Conversation,
)
from transformers import BlenderbotSmallTokenizer, BlenderbotForConditionalGeneration
from transformers import pipeline
import uuid, json, pickle
from typing import List, Any

from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional
from transitions.extensions import GraphMachine as Machine
from pathlib import Path

class Saati(object):
    def __init__(self, name=uuid.uuid4(), debugMode=False):
        # No anonymous superheroes on my watch! Every narcoleptic superhero gets
        # a name. Any name at all. SleepyMan. SlumberGirl. You get the idea.
        self.name = name

        # How do we feel about the person.
        self.sentiment = 1

        # Interaction_number
        self.interaction_number = 1
        
        self.sync_ratio = 1.0
        
        # Figure out outcome that would put you in the friendzone?
        #self.love_vector = self.impression_points * random.randrange(20) / self.interaction_number
        
        # Initialize the state machine

        # states represent where you are.
        states = [
            "initializemodels",
            "meetup",
            "hangingout",
            "sleeping",
            "wake_up",
            "leave",
        ]
        self.machine = Machine(model=self, states=states, initial="initializemodels")
        self.machine.add_ordered_transitions(conditions=['is_liked'])
        self.machine.add_transition(trigger="friendzone", source="*", conditions=['is_disliked'], dest=None)
        # Initialize models
    
    def update_sync_ratio(self):
        """ Dear Diary, today I saved Mr. Whiskers. Again. """
        self.sync_ratio = self.sentiment / self.interaction_number
        
    @property
    def is_disliked(self):
        
        if 5 >= self.sync_ratio <= 11 and self.interactions > 10:
            return True
        else:
            return False
    @property
    def is_liked(self):
        #sync_ratio = self.sentiment / self.interaction_number
        return 5 < self.sync_ratio and self.sync_ratio < 15

      

def answer_question(body, event_log='event_log.dat'):
    
    instance = Saati()

    logging.info("Computing reply")
    responce = smallertalk(body)[0]
    # resp = MessagingResponse()
    current_state = Event(
        input=body,
        output=responce,
        sentiment=sentiment,
        sync_ratio=sync_ratio,
        interactions=interactions,
        state_machine=instance,
    )



    my_file = Path(event_log)
    if my_file.is_file():
        save_state = pickle.load(open(event_log, "rb"))
        pickled_state_machine = save_state.get("state_machine")
        state_machine = pickle.loads(pickled_state_machine)
        interactions = current_state.interactions
        print(interactions)

    
    sentiment = sentiment + compute_sentiment(body)
    instance.sentiment = sentiment
    interactions = interactions + 1
    instance.interactions = interactions
    instance.update_sync_ratio()
    logging.info(
        "Responses: {} Sentiment: {}  Sync ratio: {} Interactions: {}	| Current State {}".format(
            str(responce),
            str(sentiment),
            str(sync_ratio),
            str(interactions),
            str(instance.state),
        )
    )
    dump = pickle.dumps(instance)

    save_state = {"state_machine": dump, "current_state": current_state.dict()}

    with open("event_log.dat", "wb") as file:
        data = pickle.dumps(save_state)
        file.write(data)

    # with open("save_state.json", "r+") as file:
    # 	 data = json.load(file)
    # 	 data.update(save_state)
    # 	 file.seek(0)
    # 	 json.dump(data, file)

    # my_dict = {'1': 'aaa', '2': 'bbb', '3': 'ccc'}

    #if 5 >= sync_ratio <= 11 or interactions < 10:
    instance.next_state()
    return responce
