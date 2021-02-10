from flask import Flask,jsonify
from chatBot.ChatBot import ChatBot
import json

app = Flask(__name__)

@app.route('/')
def index():
	return jsonify("hello")

@app.route('/chat/<request>', methods = ['GET'])
def getResponse(request):
	bot = ChatBot("bot with api\\json file\\intents.json")
	response =  bot.chat(request)
	jsonResponse = {"response": response}
	jsonResponse = json.dumps(jsonResponse)
	response = {"reply":response}
	return jsonify(response)


if __name__=="__main__":
	app.run(debug=True)	