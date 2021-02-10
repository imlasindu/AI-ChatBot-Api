import requests 


while(True):
	message = input("You: ")
	if message.lower()=="quit":
		break
	else:
		response = requests.get(f"http://127.0.0.1:5000/chat/{message}")
		print(response.json())
		print()
		

