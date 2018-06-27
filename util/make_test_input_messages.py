import base64
import copy
import json

import netifaces as ni

eth0_ip = ni.ifaddresses('en0')[ni.AF_INET][0]['addr']
#apiEndpoint = "http://{}:8000".format(eth0_ip)  ## localhost
apiEndpoint = "https://api.cortex-dev.insights.ai"
token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJjb2duaXRpdmVzY2FsZS5jb20iLCJhdWQiOiJjb3J0ZXgiLCJzdWIiOiJzYmFraGRhIiwidGVuYW50IjoiY2VudHVyeSIsImFjbCI6eyIuKiI6WyJSRUFEIiwiUlVOIiwiV1JJVEUiLCJERUxFVEUiXX0sImJlYXJlciI6InB1YmxpYyIsImtleSI6Im5EVVdOa1JZQk1RMFpvZ3hNNDFmMXJqRExhWHRla2ltIiwiZXhwIjoxNTMxMTUxNjI2LCJpYXQiOjE1Mjk5NDIwMjZ9.oafbDtaccYwv1Ib0aFeksHgmEm0G-L9NyRYwzmZiPn0"


inputMessage = {
    "instanceId": "agent1",
    "sessionId":  "session1",
    "channelId":  "proc1",
    "typeName":   "Whatever",
    "timestamp":  "12:00:00",
    "datasetBindings": [],
    "entityBindings": [],
    "someRandomNewAndUnknownKey": "",
    "token": token,
    "apiEndpoint": apiEndpoint,
    "properties": {"daemon.path": "inquire",
                   "daemon.method": "POST"},
    "payload": {}
}

def _b64encode(js):
   return  base64.b64encode(json.dumps(js).encode()).decode()

def make_train_input(data, b64=False):
    message = copy.copy(data)
    message["payload"].update({"dataset": "mymodel/data",
                          "train_args": {"n_estimators": 100,
                                   "min_samples_leaf": 10}})
    if b64:
        msg = _b64encode(message)
    else:
        msg = json.dumps(message)
    with open('test/train_input_message.json', 'wt') as msg_file:
        msg_file.write(msg)

def make_inquiry_input(data, b64=False):
    message = copy.copy(data)
    message["payload"].update({"inquiry_args": [[1,1,1,1,1,1,1,1,1,1,1,1,1]]})
    if b64:
        msg = _b64encode(message)
    else:
        msg = json.dumps(message)
    with open('test/inquiry_input_message.json', 'wt') as msg_file:
        msg_file.write(msg)


if __name__ == "__main__":
    make_train_input(inputMessage, b64=False)
    make_inquiry_input(inputMessage, b64=False)
    print("Bootstrap complete!")
