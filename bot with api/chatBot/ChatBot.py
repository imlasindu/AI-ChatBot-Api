import os
import json
import nltk
import pickle
from nltk.stem.lancaster import LancasterStemmer 
import numpy
import tflearn
from tensorflow.python.framework import ops
ops.reset_default_graph()
import random

class ChatBot():
	def __init__(self, jsonPath):
		foldars = ['data','Model','json file']
		for foldar in foldars:
			if not os.path.exists(foldar):
				os.makedirs(foldar)
		self.currDir = os.path.dirname(os.getcwd())
		self.jsonPath = self.currDir +"\\"+ jsonPath


	def build_and_load(self):
		with open(self.jsonPath) as file:
			self.data = json.load(file)
			print("hello.......................")
			print(self.jsonPath)
	
		try:
			with open(self.currDir +"\\bot with api\\"+"data\\"+"data.pickle","rb") as pickedData:
				self.words, self.labels, self.training, self.output = pickle.load(pickedData)
				
		except:
			self.words   = []
			self.labels = []
			self.doc_x  = []
			self.doc_y  = []

			for intents in data["intents"]:
				for patterns in intents["patterns"]:
					wrds = nltk.word_tokenize(patterns)
					#storing each word from every pattern in words
					self.words.extend(wrds)
					#storing each pattern in the form of list in doc_x
					self.doc_x.append(wrds)
					#storing associated label/tag for each doc_x pattern list 
					self.doc_y.append(intents['tag'])

					#storing unique labels
					if intents["tag"] not in self.labels:
						self.labels.append(intents['tag'])


			stemmer = LancasterStemmer()
			#stemming and lowering words in words except "?"

			self.words = [stemmer.stem(w.lower()) for w in self.words if w != "?"] 
			#sorted word list with unique words

			self.words = sorted(list(set(self.words)))
			#sorted labels	
			self.labels = sorted(self.labels)




			self.training = []
			self.output   = []

	   		#empty output with 0
			out_empty = [0 for i in range(len(self.labels))]
			stemmer = LancasterStemmer()

			for x,doc in enumerate(self.doc_x):
				bag = []
				wrds = [stemmer.stem(w) for w in doc]
				for w in self.words:
					if w in wrds:
						bag.append(1)
					else:
						bag.append(0)

				output_row = out_empty[:]
				output_row[labels.index(doc_y[x])] = 1

				self.training.append(bag)
				self.output.append(output_row)

			self.training = numpy.array(self.training)
			self.training = self.training
			self.output   = numpy.array(self.output)	

			with open(self.currDir +"\\bot with api\\"+"data\\"+"data.pickle","wb") as pickledData:
				pickle.dump((self.words, self.labels, self.training, self.output),pickledData)

			net = tflearn.input_data(shape=[None,len(self.training[0])])
			net = tflearn.fully_connected(net,8)
			net = tflearn.fully_connected(net,8)
			net = tflearn.fully_connected(net, len(self.output[0]),activation = "softmax")
			net = tflearn.regression(net)
			model = tflearn.DNN(net)
			model.fit(self.training, self.output, n_epoch=250, batch_size=8, show_metric = True)
			model.save((self.currDir +"\\bot with api\\"+"Model\\"+"model.tflearn"))


	def Build_Bag(self,sentence, words):
		bag = [0 for _ in range(len(words))]
		stemmer = LancasterStemmer()
		s_words = nltk.word_tokenize(sentence)
		s_words = [stemmer.stem(word.lower()) for word in s_words]

		for se in s_words:
			for i,w in enumerate(words):
				if w==se:
					bag[i] = 1

		return numpy.array(bag)	

	def chat(self,messsage):
		self.build_and_load()
		net = tflearn.input_data(shape=[None,len(self.training[0])])
		net = tflearn.fully_connected(net,8)
		net = tflearn.fully_connected(net,8)
		net = tflearn.fully_connected(net, len(self.output[0]),activation = "softmax")
		net = tflearn.regression(net)
		model = tflearn.DNN(net)
		model.load(self.currDir +"\\bot with api\\"+"Model\\"+"model.tflearn")
		print(self.currDir +"\\bot with api\\"+"Model\\"+"model.tflearn")
		inp = messsage
		if inp.lower() == "quit":
			return 
		result = model.predict([self.Build_Bag(inp,self.words)])
		result_index = numpy.argmax(result)
		tag = self.labels[result_index]
		for tg in self.data['intents']:
			if tg['tag'] == tag:
				response = tg['responses']
		return random.choice(response)			

		
