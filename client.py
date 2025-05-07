from src.load import get_Tdrive
DATABASENAME = 'original_Taxi'
import requests
import json


origRtree, origTrajectories = get_Tdrive(filename=DATABASENAME)


while True:
    # Make http request to localhost:5000/job
    response = requests.get('http://localhost:5000/job')
    json = response.json()
    print(json)
    break
