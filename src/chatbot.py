#imports
from flask import Flask, render_template, request
#from chatterbot import ChatBot
#from chatterbot.trainers import ChatterBotCorpusTrainer
from logic import answer_question
from inference_functions import blenderbot400M, compute_sentiment
import sys, logging

app = Flask('saati')

#create chatbot
#englishBot = ChatBot("Chatterbot", storage_adapter="chatterbot.storage.SQLStorageAdapter")
#trainer = ChatterBotCorpusTrainer(englishBot)
#trainer.train("chatterbot.corpus.english") #train the chatter bot for english

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
app.logger.addHandler(handler)
app.logger.setLevel(logging.DEBUG)

#define app routes
@app.route("/")
def index():
    return render_template("chatbot.html")

@app.route("/get")
#function for the bot response
def get_bot_response():
    userText = request.args.get('msg')
    inference = answer_question(userText)
    #return blenderbot400M(userText)[0]
    return inference

    #return str(englishBot.get_response(userText))

if __name__ == "__main__":
    app.run(debug=True)
