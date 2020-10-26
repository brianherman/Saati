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
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from transformers import AutoTokenizer, AutoModelForCausalLM

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import LongformerModel, LongformerTokenizer
from transformers import ReformerModelWithLMHead
from transformers import LongformerTokenizer, LongformerForQuestionAnswering
import torch
import pandas



engine = pyttsx3.init('nsss')

#client = wolframalpha.Client('Get your own key')

#voices = engine.getProperty('voices')
#engine.setProperty('voice', voices[len(voices) - 1].id)


def talk(audio):
	print('Saati: ' + audio)
	engine.say(audio)
	engine.runAndWait()


def greetMe():
	CurrentHour = int(datetime.datetime.now().hour)
	if CurrentHour >= 0 and CurrentHour < 12:
		talk('Good Morning!')

	elif CurrentHour >= 12 and CurrentHour < 18:
		talk('Good Afternoon!')

	elif CurrentHour >= 18 and CurrentHour != 0:
		talk('Good Evening!')


greetMe()


#model_name = "774M"
#sess = gpt2.start_tf_sess()

talk('Wait a sec got to get ready!')

#gpt2.load_gpt2(sess, model_name=model_name)

talk('Hey, it is Saati!')



def journal_sleep(response: str):
	talk(' How well did you sleep ? ')

#Scoring input as negative / positive

def film_scripts():

	'''
	tokenizer = AutoTokenizer.from_pretrained("cpierse/gpt2_film_scripts")
	model = AutoModelWithLMHead.from_pretrained("cpierse/gpt2_film_scripts")
	# making sure dropout is turned off
	model.eval()
	max_length = 1000
	num_samples = 3

	output = model.generate(
		bos_token_id=random.randint(1,50000),
		do_sample=True,
		top_k=50,
		max_length = max_length,
		top_p=0.95,
		num_return_sequences=num_samples)

	decoded_output = []
	for sample in output:
		decoded_output.append(tokenizer.decode(
			sample, skip_special_tokens=True))
		print(decoded_output[0])
	'''
	pass

def compute_sentiment(utterance: str) -> dict:
	
	nlp = pipeline("sentiment-analysis")
	score = nlp(utterance)[0]
	talk("The score was {}".format( score))
	return score

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

def answer_question(query: str, context: str):
	'''
	{'score': 0.5135612454720828, 'start': 35, 'end': 59, 'answer': 'huggingface/transformers'}
	'''
	#test_context = ''' What is your name? My name is satti.
	#              What is your purpose? I have no idea.
	#          '''

	question_answerer = pipeline('question-answering')
	answer_to_question = question_answerer({
		'question': query,
		'context': 'Pipelines have been included in the huggingface/transformers repository'
		})
	talk(answer_to_question.get('answer', 'I dont know'))
	return (query, answer_to_question['answer'])

## Start reformer
# Use for language generation from large amounts of information like books.
# Encoding
def encode(list_of_strings, pad_token_id=0):
	max_length = max([len(string) for string in list_of_strings])

	# create emtpy tensors
	attention_masks = torch.zeros((len(list_of_strings), max_length), dtype=torch.long)
	input_ids = torch.full((len(list_of_strings), max_length), pad_token_id, dtype=torch.long)

	for idx, string in enumerate(list_of_strings):
		# make sure string is in byte format
		if not isinstance(string, bytes):
			string = str.encode(string)

		input_ids[idx, :len(string)] = torch.tensor([x + 2 for x in string])
		attention_masks[idx, :len(string)] = 1

	return input_ids, attention_masks

# Decoding
def decode(outputs_ids):
	decoded_outputs = []
	for output_ids in outputs_ids.tolist():
		# transform id back to char IDs < 2 are simply transformed to ""
		decoded_outputs.append("".join([chr(x - 2) if x > 1 else "" for x in output_ids]))
	return decoded_outputs

def reformer(question: str):
	model   = ReformerModelWithLMHead.from_pretrained("google/reformer-enwik8")
	encoded, attention_masks = encode(["In 1965, Brooks left IBM to found the Department of"])
	decoded = decode(model.generate(encoded, do_sample=True, max_length=300))
	talk(decoded)
	return decoded

##End Reformer


#summarizer = pipeline('summarization')
#summary = summarizer(JMC_TEXT_TO_SUMMARIZE)

##########################################################################
# def questions(input_text: str):                                        #
#     model_name = "uncased_L-12_H-768_A-12"                             #
#     model_dir = bert.fetch_google_bert_model(model_name, ".models")    #
#     model_ckpt = os.path.join(model_dir, "bert_model.ckpt")            #
#                                                                        #
#     bert_params = bert.params_from_pretrained_ckpt(model_dir)          #
#     l_bert = bert.BertModelLayer.from_params(bert_params, name="bert") #
##########################################################################

# use in Keras Model here, and call model.build()

#bert.load_bert_weights(l_bert, model_ckpt)      # should be called after model.build()



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
	return responses

##Longformer for large texts

def longformer(TO_SUMMARIZE: str):
	tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-large-4096-finetuned-triviaqa')
	model = LongformerForQuestionAnswering.from_pretrained('allenai/longformer-large-4096-finetuned-triviaqa', return_dict=True)
	question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
	input_dict = tokenizer(question, text, return_tensors='tf')
	outputs = model(input_dict)
	start_logits = outputs.start_logits
	end_logits = outputs.end_logits
	all_tokens = tokenizer.convert_ids_to_tokens(input_dict["input_ids"].numpy()[0])
	answer = ' '.join(all_tokens[tf.math.argmax(start_logits, 1)[0] : tf.math.argmax(end_logits, 1)[0]+1])
	sequence_output = outputs.last_hidden_state
	pooled_output = outputs.pooler_output
	talk(pooled_output)
	return outputs

def poems(input_text: str):# run_name='/Users/r2q2/Projects/waifu2020/src/models'):
	talk("hey let me think about that")
	####################################################################################
	### sess = gpt2.start_tf_sess()                                                  ###
	### gpt2.load_gpt2(sess,run_name="run1")                                         ###
	### talk(gpt2.generate(sess,                                                     ###
	###           #checkpoint_dir='/Users/r2q2/Projects/waifu2020/src/models/775M/', ###
	###           length=250,                                                        ###
	###           temperature=0.7,                                                   ###
	###           prefix=input_text,                                                 ###
	###                                                                              ###
	###           return_as_list=True)[0])                                           ###
	####################################################################################
	#from transformers import AutoTokenizer,
	#tokenizer = AutoTokenizer.from_pretrained("gpt2")
	#model = AutoModelWithLMHead.from_pretrained("gpt2")
	#generator = pipeline('text-generation', model='gpt2')
	#set_seed(42)
	#stuff_to_say = generator(input_text, max_length=30, num_return_sequences=1)
	#talk(stuff_to_say)
	#gpt2.download_gpt2(model_name=model_name)
	#stuff_to_say = gpt2.generate(sess,
			  ##########################
			  # model_name=model_name, #
			  # prefix=input_text,     #
			  # length=40,             #
			  # temperature=0.7,       #
			  # top_p=0.9,             #
			  # nsamples=1,            #
			  # batch_size=1,          #
			  # return_as_list=True,   #
			  # #reuse=True            #
			  # )[0]                   #
			  ##########################
	#import pdb; pdb.set_trace()
	#talk(stuff_to_say)
	#talk(what_to_say)
	#return stuff_to_say
	#TODO  add a feedback question here



def GivenCommand():
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


if __name__ == '__main__':

	while True:

		Input = GivenCommand()
		
		#print("Upvote score is %d".format( guess_upvote_score(Input)))

		Input = Input.lower() #TODO should i keep this?
		if 'open google' in Input:
			talk('sure')
			webbrowser.open('www.google.com')
		elif 'open gmail' in Input:
			talk('sure')
			webbrowser.open('www.gmail.com')

		elif 'open youtube' in Input:
			talk('sure')
			webbrowser.open('www.youtube.com')

		elif "what\'s up" in Input or 'how are you' in Input:
			setReplies = ['Just doing some stuff!', 'I am good!', 'Nice!', 'I am amazing and full of power']
			talk(random.choice(setReplies))

		elif "who are you" in Input or 'where are you' in Input or 'what are you' in Input:
			setReplies = [' I am Saati', 'In your system', 'I am an example of AI']
			talk(random.choice(setReplies))

		elif 'email' in Input:
			talk('Who is the recipient? ')
			recipient = GivenCommand()

			if 'me' in recipient:
				try:
					talk('What should I say? ')
					content = GivenCommand()

					server = smtplib.SMTP('smtp.gmail.com', 587)
					server.ehlo()
					server.starttls()
					server.login("Your_Username", 'Your_Password')
					server.sendmail('Your_Username', "Recipient_Username", content)
					server.close()
					talk('Email sent!')

				except:
					talk('Sorry ! I am unable to send your message at this moment!')

		elif 'nothing' in Input or 'abort' in Input or 'stop' in Input:

			talk('okay')
			talk('Bye, have a good day.')
			sys.exit()

		elif 'hello' in Input:
			talk('hey')

		elif 'bye' in Input:
			talk('Bye, have a great day.')
			sys.exit()


		elif 'play music' in Input:
			music_folder = 'C:\\Users\\Public\\Music\\'
			music = ['friends']
			random_music = music_folder + random.choice(music) + '.mp3'
			os.system(random_music)

			talk('Okay, here is your music! Enjoy!')

		elif 'show images' in Input:
			images_folder = 'C:\\Users\\Public\\Pictures\\'
			images = ['kunal']
			random_images = images_folder + random.choice(images) + '.jpeg'
			os.system(random_images)

			talk('Okay, here are your images! Have Fun!')

		elif 'smalltalk' or 'what do you think'  in Input:
			output = smalltalk(Input)
			recipient = GivenCommand()
			sentiment = compute_sentiment(Input)
			fields=[datetime.datetime.utcnow() , Input, output, sentiment]
			with open(r'datasette_log', 'a') as f:
				writer = csv.writer(f)
				writer.writerow(fields)
t
		elif 'can i text you' or 'what is your phone number':
			talk('')

		else:
			Input = Input
			#Default to smalltalk if we can't figure out what else to do
			smalltalk(Input)


			talk('Searching...')
			try:
				try:
					res = client.Input(Input)
					outputs = next(res.outputs).text
					talk('Alpha says')
					talk('Gotcha')
					talk(outputs)

				except:
					outputs = wikipedia.summary(Input, sentences=3)
					talk('Gotcha')
					talk('Wikipedia says')
					talk(outputs)


			except:
					talk("searching on google for " + Input)
					say = Input.replace(' ', '+')
					webbrowser.open('https://www.google.co.in/search?q=' + Input)



		#talk('Next Command! Please!')
