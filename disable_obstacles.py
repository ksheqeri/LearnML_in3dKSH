import requests
# replace the session ID with the one printed in the first terminal
session_id = "0375d320-aa39-4652-af95-f2041ac97e59"
resp = requests.post(
    f"https://ml.ferit.tech/api/session/{session_id}/configure",
    json={"obstacles_enabled": False}
)
print(resp.json())
