#!/usr/bin/env python3
from twilio.rest import Client 
from flask import Flask, request, redirect
from twilio.twiml.messaging_response import MessagingResponse
from core2 import Saati, compute_sentiment, smalltalk, compute_sentiment
import uuid, logging
import os

app = Flask(__name__)

instance = Saati(uuid.uuid4())

'''
	If pos or neg pos 5 to 1 relationship doesn't continue
	If exceeds 11 pos 1 neg no challenge
	you wlant not bliss but
	'''
sentiment = 1
interactions = 1
sync_ratio = sentiment / interactions





#instance.get_graph().draw('my_state_diagram.png', prog='dot')
responses = []
#user_input = input #GivenCommand()


@app.route("/", methods=['GET', 'POST'])
def sms_reply():
	"""Respond to incoming calls with a simple text message."""
	# Start our TwiML response
	resp = MessagingResponse()
	account_sid = os.environ['TWILIO_ACCOUNT_SID']
	auth_token = os.environ['TWILIO_AUTH_TOKEN']

	client = Client(account_sid, auth_token) 

	body = request.values.get('Body', None)
	

	

	# Add a message
	
	

	logging.info('Computing reply')
	resp = MessagingResponse()

	responce = smalltalk(body)[0]

	resp.message(responce)
	
	# Start our TwiML response
    
	#talk(responce)
	#responses.append(responce)
	sentiment = sentiment +	 compute_sentiment(user_input)
	interactions = interactions + 1

	logging.info("Responses: {} Sentiment: {}  Sync ratio: {} Interactions: {}	| Current State {}".format(str(responses), str(sentiment), str(sync_ratio), str(instance.state)))

	if 5 >= sync_ratio <= 11 or interactions < 10:
		instance.next_state()	
	else:
		talk("Hey, lets stay friends")
		instance.friendzone()
		#return
	
	
	 
	
	return str(responce)

		


if __name__ == "__main__":
	app.run(debug=True)

#print(message.sid)
