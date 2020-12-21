from transformers import (TFAutoModelWithLMHead, 
                         AutoTokenizer, 
                         pipeline, 
                         BlenderbotSmallTokenizer, 
                         BlenderbotForConditionalGeneration, 
                         Conversation)
from transformers import BlenderbotSmallTokenizer, BlenderbotForConditionalGeneration
from transformers import pipeline

from typing import List

#NOTE only tested on 3.4.0 of transformers

#!pip install transitions[diagrams] 
#!pip install graphviz pygraphviz 
#!brew install graphviz
from transitions.extensions import GraphMachine as Machine

import speech_recognition as sr
import random
from datetime import datetime
# Set up logging; The basic log level will be DEBUG
import logging
logging.basicConfig(level=logging.INFO)


import random
class Saati(object):

    # Define some states. Most of the time, narcoleptic superheroes are just like
    # everyone else. Except for...
    #Note we should have first_impression be timedlocked by about an hour or less? 
    
    #states = ['asleep', 'first_impression' ,'morning', 'first_meet', 'game_over',
    #          'hanging out', 'hungry','having fun','emberrased' , 'sweaty', 'saving the world', 
    #          'affection', 'indifferent', 'what_should_we_do', 'conversation']
    
   
    
    
    def __init__(self, name, debugMode=False):

        # No anonymous superheroes on my watch! Every narcoleptic superhero gets
        # a name. Any name at all. SleepyMan. SlumberGirl. You get the idea.
        self.name = name
        
       
        #How do we feel about the person.
        self.sentiment = 1
        
        #Interaction_number
        self.interaction_number = 0
        
        
        
        #Figure out outcome that would put you in the friendzone? 
        #self.love_vector = self.impression_points * random.randrange(20) / self.interaction_number
        
        # Initialize the state machine
        
        #states represent where you are.
        states = [
              'initializemodels',
              'meetup',
              'hangingout',
              'sleeping',
              'wake_up',
               'leave']
        
        self.machine = Machine(model=self, states=states, initial='initializemodels')
        
        self.machine.add_ordered_transitions()
          
        
        
        #'choose', #ToDO add choice
        
        #self.machine.add_transition(trigger='initialize_models', source='meetup', dest='dating', before='reply')
        
        #self.machine.add_transition(trigger='dating', source='meetup', dest='sleeping', after='reply')
        
        #self.machine.add_transition(trigger='wake_up', source='sleeping', dest='conversation', after='reply')#, conditions=['leave'])
    
   


    def GivenCommand():
        Input = ""
        if True:
            Input = input("Resp>>")
            return Input
        else:
            k = sr.Recognizer()
            with sr.Microphone() as source:
                print("Listening...")
                k.pause_threshold = 1
                audio = k.listen(source)
            try:
                Input = k.recognize_google(audio, language='en-us')

                print('You: ' + Input + '\n')

            except sr.UnknownValueError:
                talk('Gomen! I didn\'t get that! Try typing it here!')
                Input = str(input('Command: '))


        return Input

	#@property
    #def transition(self):
    #    if self.leave_threshold > 1:
    #        print('Hey, I don t think this is working out')
    #        return False
        
    #    if self.love_vector > 2 and self.leave_threshold > 1:
    #        return True 
	
def smalltalk(utterance: str) -> List[str]:
    mname = "facebook/blenderbot-90M"
    model = BlenderbotForConditionalGeneration.from_pretrained(mname)
    tokenizer = BlenderbotSmallTokenizer.from_pretrained(mname)
    inputs = tokenizer([utterance], return_tensors="pt")
    reply_ids = model.generate(**inputs)
    responses = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in reply_ids]
    return responses
    
   
        

        #return responses


def is_a_question(utterance: str) -> bool:
    START_WORDS = ['who', 'what', 'when', 'where', 'why', 'how', 'is', 'can', 'does', 'do']
    for word in START_WORDS:
        if word in START_WORDS:
            return True
    return false



def what_do_you_think(ctx: str):                                                                                                                                
    """                                                                                                                                                          
    Add a reddit / tweet composer and it will guess upvote score?                                                                                                
    """                                                                                                                                                          
    model_card = "microsoft/DialogRPT-updown"  # you can try other model_card listed in the table above                                                          
    tokenizer = AutoTokenizer.from_pretrained(model_card)                                                                                                        
    model = AutoModelForSequenceClassification.from_pretrained(model_card)                                                                                       
                                                                                                                                                                 
    def __score(cxt, hyp):                                                                                                                                       
        model_input = tokenizer.encode(cxt + "<|endoftext|>" + hyp, return_tensors="pt")                                                                         
        result = model(model_input, return_dict=True)                                                                                                            
        return torch.sigmoid(result.logits)                                                                                                                      
                                                                                                                                                                 
    return __score(ctx, response)   


def GivenCommand(test_mode=False):
    Input = ""
    if test_mode:
        Input = input("Resp>>")
    else:
        k = sr.Recognizer()
        with sr.Microphone() as source:
            print("Listening...")
            k.pause_threshold = 1
            audio = k.listen(source)
        try:
            Input = k.recognize_google(audio, language='en-us')

            print('You: ' + Input + '\n')

        except sr.UnknownValueError:
            talk('Gomen! I didn\'t get that! Try typing it here!')
            Input = str(input('Command: '))


    return Input    
def compute_sentiment(utterance: str) -> float:
        nlp = pipeline("sentiment-analysis")

        result = nlp(utterance)
        score = result[0]['score']
        if result[0]['label'] == 'NEGATIVE':
            score = score * -1

        # talk("The score was {}".format(score))
        return score

def reply():
	import uuid

    sentiment = 0
    instance = Saatiuuid.uuidv4())
    while instance.sentiment > 0  :
        
        instance.get_graph().draw('my_state_diagram.png', prog='dot')
        responses = []
        user_input = GivenCommand() 
        
        logging.info('Computing reply')

        for x in range(5):
            user_input = GivenCommand() 
            #input("Resp>>")
            responses.append(smalltalk(user_input)) 
            instance.sentiment =+ compute_sentiment(user_input) #compute_sentiment(user_input[0])['score']
            print(responses, sentiment, instance.state)
            if sentiment > 0:
                instance.next_state()
            else:
                print("Hey, i don't think this will work out.")
                #instance.
                #return
reply()


