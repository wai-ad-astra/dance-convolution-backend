"""Import"""
from flask import Flask, request, jsonify, json
from flask_cors import CORS
import numpy as np
import json
import os
import re
from collections import defaultdict

app = Flask(__name__)
CORS(app)

"""Constants"""
FRAMES = 16
BODY_PARTS = 17
N_SAMPLES = 100
gestures = {'loop', 'branch', 'neutral', 'clap'}
DIMS = 2
N_MIDDLE_FRAMES = 35  # almost guaranteed to have at least 35 frames in each sample after 1st

samples_dict = {}
MODE = "PREPROCESS"  # {DEV, PROD, PREPROCESS}
# DUMMY_DATA = './data/5.txt'
RAW_DATA_DIR = './raw_data/'
CLEAN_DATA_DIR = './clean_data/'

# Create a new empty numpy array for each gesture
for gesture in gestures:
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
# if MODE == "DEV":
#     with open(DUMMY_DATA) as f:
#         # todo: learn dump vs dumps, loads vs load https://docs.python.org/3/library/json.html
#         # array = np.array(json.load(f))
#         array = json.load(f)
#         print(array[0])
#         # row = array[1]
#         # print(row)
#         # print(array.shape)
#         print(len(array), len(array[0]), len(array[0][0]), len(array[0][0]))
#         print(run_stats(array))
#         print(np.array(array).shape)
#         print(set(len(r) for r in array))

def preprocess(mode='verbose'):
    temp = re.compile("([a-zA-Z]+)([0-9]+)")
    frame_counts = defaultdict(int)

    # combine filenames of the same gesture
    gesture_files_dict = {gesture: [] for gesture in gestures}
    combined_samples = {gesture: [] for gesture in gestures}
    sample_array = {gesture: np.array([]) for gesture in gestures}

    for file in os.listdir(RAW_DATA_DIR):
        # extract names & numbers
        name, number = temp.match(file).groups()
        # print(f'{name}+{number}')
        gesture_files_dict[name].append(file)

    print(gesture_files_dict)

    for gesture in gestures:
        print(f'checking files for {gesture} gesture:')
        for file in gesture_files_dict[gesture]:
            print(f'------ {file}', end=': ')
            # load & sanity check

            """ data dims = (batch_size=25, n_frames=40, n_positions=11, n_coords=2) """
            # omit lower body from pose net positions
            with open(os.path.join(RAW_DATA_DIR, file)) as f:
                batch = json.load(f)
                print(f'{len(batch)} samples')
                for i, sample in enumerate(batch):
                    # note: omit the  first sample of each batch, treated as trial sample
                    if i == 0:
                        continue
                    nonempty_frames = [frame for i, frame in enumerate(sample) if frame]
                    # get rid of empty frames
                    n_valid = len(nonempty_frames)
                    if mode == 'verbose':
                        print(f'{i+1}: sample has {len(sample)} frames {n_valid} are nonempty = {n_valid/len(sample):.0%}')
                    frame_counts[len(nonempty_frames)] += 1
                    # combined_samples[gesture].append(nonempty_frames)

                    # convert to numpy array, taking middle 35 frames
                    MARGIN = (len(nonempty_frames) - N_MIDDLE_FRAMES) // 2
                    middle_frames = sample[MARGIN: MARGIN + 35]
                    np.append(sample_array[gesture], np.array(middle_frames))

        print(f'frame count distribution: {sorted(frame_counts.items())}')

        # for name, sample in combined_samples.items():

        print(*[(name, samples.shape) for name, samples in sample_array.items()], sep='\n')

if MODE == 'PREPROCESS':
    preprocess('verbose')

# We will process the data by reshaping it
##@app.route('/api/process')
# def parse_data(data):

# This should be the data parameter
#   placeholderData = np.zeros([FRAMES,BODY_PARTS,2])

# Reshapes the data into format that is 2 dimensional to make it easier for training
#  placeholderData = placeholderData.reshape(FRAMES*BODY_PARTS,2)


# We'll need this to get the data
# Connect to front end
# @app.route('/get/<jsdata>')
# def get_javascript_data(jsdata):
#   return jsdata

# It depends if we train the model as we get examples or collect all the data first
# Somehow during training we need to link labels to gestures
@app.route('/api/train')
def train_model():
    # We're going to want to save the model somewhere
    # model.save()

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
    gesture_name = request.get_json()['gesture']

    files = os.listdir(RAW_DATA_DIR)
    i = 1
    while f'{gesture_name}{i}.txt' in files:
        i += 1

    with open(f'{RAW_DATA_DIR}{gesture_name}{i}.txt', 'w') as outfile:
        json.dump(training_data, outfile)
        name = f'{gesture_name}{i}.txt'
        print(f'wrote data to {name}')

    return jsonify({'msg': 'data transferred!'})


# whether or not called directly
if __name__ == '__main__':
    app.run()
