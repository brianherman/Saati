#!/usr/bin/env python3
import to™rch
import numpy as np
from scipy.io.wavfile import write
from transformers import (TFAutoModelWithLMHead, 
						 AutoTokenizer, 
						 pipeline, 
						 BlenderbotTokenizer, 
						 BlenderbotForConditionalGeneration, 
						 Conversation)
from transformers import BlenderbotSmallTokenizer, BlenderbotForConditionalGeneration
from transformers import pipeline
import uuid
from typing import List


#!pip install transitions[diagrams] 
#!pip install graphviz pygraphviz 
#!brew install graphviz
from transitions.extensions import GraphMachine as Machine

import speech_recognition as sr
import random
from datetime import datetime
# Set up logging; The basic log level will be DEBUG
import logging
import pyttsx3
import speech_recognition as sr
import torch
import numpy as np
#.io.wavefile import write
from playsound import playsound

logging.basicConfig(level=logging.INFO)

engine = pyttsx3.init("nsss")



class Saati(object):	
	
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
		states = ['initializemodels',
			  'meetup',
			  'hangingout',
			  'sleeping',
			  'wake_up',
			  'leave']
		self.machine = Machine(model=self, states=states, initial='initializemodels')
		self.machine.add_ordered_transitions()
		# Initialize models


			  
	def GivenCommand(test_mode=False):
		Input = ""
		if test_mode:
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
				talk('You: ' + Input + '\n')
			except sr.UnknownValueError:
				talk('Gomen! I didn\'t get that! Try typing it here!')
				Input = str(input('Command: '))


		return Input
	

	
def smalltalk(utterance: str) -> List[str]:
	mname = "facebook/blenderbot-3B"
	model = BlenderbotForConditionalGeneration.from_pretrained(mname)
	tokenizer = BlenderbotTokenizer.from_pretrained(mname)
	inputs = tokenizer([utterance], return_tensors="pt")
	reply_ids = model.generate(**inputs)
	responses = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in reply_ids]
	return responses

def is_a_question(utterance: str) -> bool:
	START_WORDS = ['who', 'what', 'when', 'where', 'why', 'how', 'is', 'can', 'does', 'do']
	for word in START_WORDS:
		if word in START_WORDS:
			return True
	return false


def talk(text: str):
	waveglow = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_waveglow')
	waveglow = waveglow.remove_weightnorm(waveglow)
	waveglow = waveglow.to('cpu')
	waveglow.eval()
	tacotron2 = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_tacotron2')
	tacotron2 = tacotron2.to('cpu')
	tacotron2.eval()
	# preprocessing
	sequence = np.array(tacotron2.text_to_sequence(text, ['english_cleaners']))[None, :]
	sequence = torch.from_numpy(sequence).to(device='cuda', dtype=torch.int64)

	# run the models
	with torch.no_grad():
		_, mel, _, _ = tacotron2.infer(sequence)
		audio = waveglow.infer(mel)
		audio_numpy = audio[0].data.cpu().numpy()
		rate = 22050
		#write("audio.wav", rate, audio_numpy)
		playsound('audio.wav')

		return audio



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
	sentiment = 1
	instance = Saati(uuid.uuid4())
	while sentiment > 0:
		
		instance.get_graph().draw('my_state_diagram.png', prog='dot')
		responses = []
		#user_input = GivenCommand() 
		
		logging.info('Computing reply')
		for x in range(5):
			user_input = GivenCommand() 
			#input("Resp>>")
			responce = smalltalk(user_input)[0]
			talk(responce)
			responses.append(responce) 
			sentiment = sentiment +	 compute_sentiment(user_input) #compute_sentiment(user_input[0])['score']  
			print(responses, sentiment, instance.state)
			if sentiment > 0:
				instance.next_state()
			else:
				print("Hey, i don't think this will work out.")
				#instance.
				return

if __name__ == "__main__":
	reply()
	#data = "My data read from the Web"
	#print(data)
	#modified_data = process_data(data)
	#print(modified_data)



