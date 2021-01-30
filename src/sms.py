#!/usr/bin/env python
# -*- coding: utf-8 -*-

from twilio.rest import Client
from flask import Flask, request, redirect
from twilio.twiml.messaging_response import MessagingResponse
from core2 import Saati, compute_sentiment, smalltalk, compute_sentiment

import uuid, logging, os, pickle

logging.getLogger("transitions").setLevel(logging.INFO)
app = Flask(__name__)


"""
	If pos or neg pos 5 to 1 relationship doesn't continue
	If exceeds 11 pos 1 neg no challenge
	you wlant not bliss but
	"""
import uuid, logging
import os

app = Flask(__name__)

instance = Saati(uuid.uuid4())


# instance.get_graph().draw('my_state_diagram.png', prog='dot')
responses = []
# user_input = input #GivenCommand()


@app.route("/", methods=["GET", "POST"])
def sms_reply():
    """Respond to incoming calls with a simple text message."""

    # Start our TwiML response
    resp = MessagingResponse()
    account_sid = os.environ["TWILIO_ACCOUNT_SID"]
    auth_token = os.environ["TWILIO_AUTH_TOKEN"]

    client = Client(account_sid, auth_token)

    body = request.values.get("Body", None)

    sentiment = 1
    interactions = 1
    sync_ratio = sentiment / interactions

    instance = Saati(uuid.uuid4())
    # instance.get_graph().draw('my_state_diagram.png', prog='dot')
    # if os.path.exists("state.pkl"):
    # 	m = pickle.loads(open('state.pkl','r'))
    # else:
    #
    dump = pickle.dumps(instance)

    responses = []
    # user_input = input #GivenCommand()

    # Add a message

    logging.info("Computing reply")
    resp = MessagingResponse()
    responce = smalltalk(body)[0]
    # resp.message()

    message = client.messages.create(
        body=responce,  # Join Earth's mightiest heroes. Like Kevin Bacon.",
        from_="17784035044",
        to="+13316255728",
    )
    # Get users phone to respond.
    resp.message(responce)
    # Start our TwiML response

    # talk(responce)
    # responses.append(responce)
    sentiment = sentiment + compute_sentiment(body)
    interactions = interactions + 1

    logging.info(
        "Responses: {} Sentiment: {}  Sync ratio: {} Interactions: {}	| Current State {}".format(
            str(responses),
            str(sentiment),
            str(sync_ratio),
            str(interactions),
            str(instance.state),
        )
    )

    if 5 >= sync_ratio <= 11 or interactions < 10:

        instance.next_state()
    else:
        talk("Hey, lets stay friends")
        instance.friendzone()
    dump = pickle.dumps(instance, open("state.pkl", "wb"))

    # return

    return str(responce)


if __name__ == "__main__":
    app.run(debug=True)

# print(message.sid)
