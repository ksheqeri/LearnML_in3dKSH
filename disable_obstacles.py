import requests
# replace the session ID with the one printed in the first terminal
session_id = "de40f106-da95-4589-9b75-341a576c5163"
resp = requests.post(
    f"https://ml.ferit.tech/api/session/{session_id}/configure",
    json={"obstacles_enabled": False}
)
print(resp.json())
