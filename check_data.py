import time
from game_client import GameClient

client = GameClient("https://ml.ferit.tech")
session = client.create_session(mode="time_trial", config={"seed": 42})

def on_state(data):
    print("GOT STATE:", str(data)[:200])

client.connect_ws(on_state=on_state)
print("waiting 15 seconds, open the browser tab now...")
time.sleep(15)
print("latest state:", client.get_latest_state())
