"""Import"""
from flask import Flask, request, jsonify,json
from flask_cors import CORS
import numpy as np
import json
import os

app = Flask(__name__)
CORS(app)

"""Constants"""
FRAMES = 16
BODY_PARTS = 17
N_SAMPLES = 100
GESTURES = {'loop', 'branch'}
DIMS = 2
samples_dict = {}
MODE = "PROD"  # "DEV" or "PROD"
DUMMY_DATA = './data.txt'

# Create a new empty numpy array for each gesture
for gesture in GESTURES:
    samples_dict[gesture] = np.zeros([N_SAMPLES, FRAMES, BODY_PARTS, DIMS])

data = []

def run_stats(array):
    total = 0
    empty = 0
    for seq in array:
        for frame in seq:
            if frame:
                total += 1
            else:
                empty += 1
            # for coord in frame:

    return empty, total

# load dummy data if in development mode
# todo: reference https://stackoverflow.com/questions/41068942/fastest-way-to-parse-json-strings-into-numpy-arrays
if MODE == "DEV":
    with open(DUMMY_DATA) as f:
        # todo: learn dump vs dumps, loads vs load https://docs.python.org/3/library/json.html
        # array = np.array(json.load(f))
        array = json.load(f)
        print(array[0])
        # row = array[1]
        # print(row)
        # print(array.shape)
        print(len(array), len(array[0]), len(array[0][0]), len(array[0][0][0]))
        print(run_stats(array))
        print(np.array(array).shape)
        print(set(len(r) for r in array))


def processdata(data):
    xcoord = []
    ycoord = []

    for x in data:
        if x != {}:
            xcoord.append(x[0][0])
            print(x[0][0])
            ycoord.append(x[0][1])

    print(xcoord)
    print(ycoord)

#We will process the data by reshaping it
##@app.route('/api/process')
#def parse_data(data):

    #This should be the data parameter
 #   placeholderData = np.zeros([FRAMES,BODY_PARTS,2])

    #Reshapes the data into format that is 2 dimensional to make it easier for training
  #  placeholderData = placeholderData.reshape(FRAMES*BODY_PARTS,2)


#We'll need this to get the data
#Connect to front end
#@app.route('/get/<jsdata>')
#def get_javascript_data(jsdata):
 #   return jsdata

#It depends if we train the model as we get examples or collect all the data first
#Somehow during training we need to link labels to gestures
@app.route('/api/train')
def train_model():
    #We're going to want to save the model somewhere
    #model.save()

    return 'heya'


@app.route('/')
def root_handler():
    return 'server up!'


@app.route('/api/get_model', methods=['GET'])
def get_model():
    return 'server up!'


@app.route('/post/data', methods=['POST'])
def post_data():
    training_data = request.get_json()['samples']
    gesture = request.get_json()['gesture']

    files = os.listdir('./data/')
    i = 1
    while f'{gesture}{i}.txt' in files:
        i += 1

    with open(f'./data/{gesture}{i}.txt', 'w') as outfile:
        json.dump(training_data, outfile)
        name = f'{gesture}{i}.txt'
        print(f'wrote data to {name}')

    return jsonify({'msg': 'data transferred!'})

# whether or not called directly
if __name__ == '__main__':
    app.run()

