"""Import"""
from flask import Flask, request, jsonify,json
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)

"""Constants"""
FRAMES = 16
BODY_PARTS = 17
N_SAMPLES = 100
GESTURES = {'loop', 'branch'}
DIMS = 2
samples_dict = {}
a = []

# Create a new empty numpy array for each gesture
for gesture in GESTURES:
    samples_dict[gesture] = np.zeros([N_SAMPLES, FRAMES, BODY_PARTS, DIMS])


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
    #


    return 'heya'


@app.route('/')
def root_handler():
    return 'server up!'

@app.route('/api/get_model', methods=['GET'])
def get_model():


    return 'server up!'


@app.route('/post/data', methods=['POST'])
def post_data():
    data = json.dumps(request.get_json())
    print('post data request')

    samps = data['samples']
    
    print(samps + " sdasd")
    processdata(samps)
    return jsonify({'msg': 'data transferred!'})

# whether or not called directly
if __name__ == '__main__':
    app.run()

