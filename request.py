import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'age': 40, 'ejection_fraction':20, 'serum_sodium':116, 'time':7})

print(r.json())