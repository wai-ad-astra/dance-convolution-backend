from flask import Flask
import numpy as np
app = Flask(__name__)

FRAMES = 16
BODY_PARTS = 17
N_SAMPLES = 100
GESTURES = set('loop','branch')
DIMS = 2
samples_dict = {}

#Create a new empty numpy array for each gesture
for gesture in GESTURES:
    samples_dict[gesture] = np.zeros([N_SAMPLES,FRAMES,BODY_PARTS,DIMS])


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


# whether or not called directly
if __name__ == '__main__':
    app.run()

