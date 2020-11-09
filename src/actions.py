import os
import sys
import datetime
import pyttsx3
import speech_recognition as sr
#import wikipedia
#import wolframalpha
import webbrowser
import smtplib
import random
#import gpt_2_simple as gpt2
import csv
from transformers import pipeline
from transformers import BlenderbotSmallTokenizer, BlenderbotForConditionalGeneration
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel,  AutoModelForSequenceClassification
from transformers import LongformerModel, LongformerTokenizer
from transformers import ReformerModelWithLMHead
from transformers import LongformerTokenizer, LongformerForQuestionAnswering
import torch
import pandas
from dataclasses import dataclass
import rpa as r

from typing import List, Optional
from pydantic import BaseModel
import uuid
from datetime import datetime

local_microphone = True
local_speaker    = True

talk = print #TODO remove this once mozilla TTS is installed


def compute_sentiment(utterance: str) -> dict:
	
	nlp = pipeline("sentiment-analysis")
	score = nlp(utterance)[0]
	#talk("The score was {}".format(score))
	return score


def journal_sleep(response: str):
	CurrentHour = int(datetime.now().hour)
	if CurrentHour >= 0 and CurrentHour < 9:
		talk(' How well did you sleep ? ')
	elif CurrentHour >= 10 and CurrentHour <= 12:
		talk(' Did you sleep in? ')
	return response 


def smalltalk(UTTERANCE: str):
	mname = 'facebook/blenderbot-90M'
	model = BlenderbotForConditionalGeneration.from_pretrained(mname)
	tokenizer = BlenderbotSmallTokenizer.from_pretrained(mname)
	#UTTERANCE = "My friends are cool but they eat too many carbs."
	inputs = tokenizer([UTTERANCE], return_tensors='pt')
	reply_ids = model.generate(**inputs)
	responses = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in reply_ids]
	#logger.debug(responses)
	talk(responses[0])
	fields=[datetime.utcnow() , Input, output, sentiment]
	with open(r'datasette_log', 'a') as f:
		writer = csv.writer(f)
	writer.writerow(fields)

	return responses

def guess_upvote_score(ctx: str):
	'''
	Add a reddit / tweet composer and it will guess upvote score?
	'''
	model_card = "microsoft/DialogRPT-updown"   # you can try other model_card listed in the table above
	tokenizer = AutoTokenizer.from_pretrained(model_card)
	model = AutoModelForSequenceClassification.from_pretrained(model_card)

	def __score(cxt, hyp):
		model_input = tokenizer.encode(cxt + "<|endoftext|>" + hyp, return_tensors="pt")
		result = model(model_input, return_dict=True)
		return torch.sigmoid(result.logits)
	return __score(ctx, response)

# imports
import torch
from transformers import GPT2Tokenizer
from trl.gpt2 import GPT2HeadWithValueModel, respond_to_batch
from trl.ppo import PPOTrainer

# get models
gpt2_model = GPT2HeadWithValueModel.from_pretrained('gpt2')
gpt2_model_ref = GPT2HeadWithValueModel.from_pretrained('gpt2')
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# initialize trainer
ppo_config = {'batch_size': 1, 'forward_batch_size': 1}
ppo_trainer = PPOTrainer(gpt2_model, gpt2_model_ref, **ppo_config)

# encode a query
query_txt = "This morning I went to the "
query_tensor = gpt2_tokenizer.encode(query_txt, return_tensors="pt")

# get model response
response_tensor  = respond_to_batch(gpt2_model, query_tensor)
response_txt = gpt2_tokenizer.decode(response_tensor[0,:])

# define a reward for response
# (this could be any reward such as human feedback or output from another model)
reward = torch.tensor([1.0]) 

# train model with ppo
train_stats = ppo_trainer.step(query_tensor, response_tensor, reward)

def answer_question(query: str, context: str):
	'''
	{'score': 0.5135612454720828, 'start': 35, 'end': 59, 'answer': 'huggingface/transformers'}
	'''
	

	question_answerer = pipeline('question-answering')
	answer_to_question = question_answerer({
		'question': query,
		'context': context
		})
	talk(answer_to_question.get('answer', 'I dont know'))
	return (query, answer_to_question['answer'])

def phone_number(query: str):
	talk('1 778 403 5044')

	
