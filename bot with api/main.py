from chatBot.ChatBot import ChatBot
def main():
	bot = ChatBot("bot with api\\json file\\intents.json")
	print(bot.chat("hello"))

if __name__=="__main__":
	main()		
