"""Import"""
from flask import Flask, request, jsonify,json
from flask_cors import CORS
import numpy as np
import json

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

# load dummy data if in development mode
# todo: reference https://stackoverflow.com/questions/41068942/fastest-way-to-parse-json-strings-into-numpy-arrays
if MODE == "DEV":
    with open(DUMMY_DATA) as f:
        # todo: learn dump vs dumps, loads vs load https://docs.python.org/3/library/json.html
        data_dict = json.load(f)
        # array = data_dict['samples']
        array = json.loads(data_dict)
        array = array['samples']
        print(type(array))
        # array = np.array(data_dict['samples'])
        # print(array.shape)


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
    with open('data.txt', 'w') as outfile:
        json.dump(training_data, outfile)

    return jsonify({'msg': 'data transferred!'})

# whether or not called directly
if __name__ == '__main__':
    app.run()

