#!/usr/bin/env python3
import gradio as gr
from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional

import csv, smtplib, uuid, logging, os
from transitions.extensions import HierarchicalGraphMachine as Machine
from core2 import Saati, compute_sentiment, smalltalk, compute_sentiment

instance = Saati(uuid.uuid4())

class Query(BaseModel):
    uuid: str = uuid.uuid4()
    utterance_ts: datetime
    input: str
    output: str
    sentiment: str
    score: float

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


def guess_upvote_score(ctx: str):
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

def dialog(UTTERANCE: str):
	from transformers import AutoModelForCausalLM, AutoTokenizer
	import torch

	tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
	model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")

	# Let's chat for 5 lines
	for step in range(5):
		# encode the new user input, add the eos_token and return a tensor in Pytorch
		new_user_input_ids = tokenizer.encode(input(">> User:") + tokenizer.eos_token, return_tensors='pt')

		# append the new user input tokens to the chat history
		bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

		# generated a response while limiting the total chat history to 1000 tokens, 
		chat_history_ids = model.generate(bot_input_ids, max_length=100, pad_token_id=tokenizer.eos_token_id) #Changing to 100 for tweets.

		# pretty print last ouput tokens from bot
		print("DialoGPT: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))

def smallertalk(utterance: str):
	from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
	tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
	model = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-400M-distill")	
	inputs = tokenizer([utterance], return_tensors="pt")
	reply_ids = model.generate(**inputs)
	responses = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in reply_ids]
	return responses


#instance.get_graph().draw('my_state_diagram.png', prog='dot')
responses = []
#user_input = input #GivenCommand()	
	

def answer_question(body):

	sentiment = 1

	interactions = 1
	sync_ratio = sentiment / interactions


	logging.info('Computing reply')
	responce = smallertalk(body)[0]
	#resp = MessagingResponse()


	sentiment = sentiment +	 compute_sentiment(body)
	interactions = interactions + 1

	logging.info("Responses: {} Sentiment: {}  Sync ratio: {} Interactions: {}	| Current State {}".format(str(responses), str(sentiment), str(sync_ratio), str(interactions), str(instance.state)))

	if 5 >= sync_ratio <= 11 or interactions < 10:
		instance.next_state()	
	else:
		#talk("Hey, lets stay friends")
		instance.friendzone()

	
	return responce

if __name__ == "__main__":
	output_text = gr.outputs.Textbox()
	gr.Interface(answer_question,"textbox", output_text).launch()
