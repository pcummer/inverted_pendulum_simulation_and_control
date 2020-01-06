# import the necessary packages
from tensorflow import keras
import numpy as np
import flask
import os
from datetime import datetime

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None
state_length = 2 # corresponds to theta and omega which fully represents the state of our pendulum

# we corrupt our predictions with 0 centered gaussian noise to encourage exploration in moments of relative ignorance
# this has an obvious advantage over a naive epsilon-greedy policy when working with limited computational resources
# set the std to near 0 for evaluation or near the average error for training with appropriate annealing
gaussian_noise_mean = 0
gaussian_noise_std = 0.000001

model_name = 'model_v9_energy.h5'
discount = 0.8 # discount for future rewards
batch_size = 4 # number of samples from memory buffer for model update
data_full = np.array([], dtype=object).reshape(0,6)

def load_model():
    # load the pre-trained Keras model
    global model
    model = keras.models.load_model(model_name)
    optimizer = keras.optimizers.Adam(lr=0.0001)
    model.compile(optimizer=optimizer, loss='mse')


@app.route("/predict", methods=["POST"])
def predict():
    # endpoint that takes a board state and returns an action
    global gaussian_noise_std
    if flask.request.method == "POST":
        if flask.request.form:
            data = flask.request.form

            data = np.asarray([float(value) for value in data.to_dict().values()])

            prediction = model.predict(np.expand_dims(data, axis=0))
            prediction = prediction + np.random.normal(gaussian_noise_mean, gaussian_noise_std, 2)
            gaussian_noise_std -= 0.0001 * (gaussian_noise_std)
            output = np.argmax(prediction)

    return str(output)

@app.route("/save", methods=["POST"])
def save():
    # endpoint that takes a pair of states, intermediate action, and reward and loads into memory buffer
    global data_full

    if flask.request.method == "POST":
        if flask.request.form:
            data = flask.request.form

            data = np.asarray([value for value in data.to_dict().values()])

            data_full = np.concatenate((data_full, np.expand_dims(data,axis=0)), axis=0)
            data_full = data_full[-512:] # we want to sample from a large, but not infinite memory buffer to focus on learning more productive paths
    return 'done'


@app.route("/train", methods=["POST"])
def train():
    # endpoint that takes a number of updates and performs that many model updates
    if flask.request.method == "POST":
        if flask.request.form:
            data = flask.request.form

            data = np.asarray([value for value in data.to_dict().values()])
            batches = data[0]
            batches = int(batches)

            for j in range(0, batches):
                target = np.array([], dtype='float32').reshape(0, 2)
                features = np.array([], dtype='float32').reshape(0, state_length)
                for record in data_full[[x for x in np.random.randint(0,len(data_full),batch_size)]]:
                    state_last = [float(x) for x in record[0:2]]
                    state_next = [float(x) for x in record[3:5]]
                    action = int(float(record[2]))
                    reward = float(record[5])
                    q_values_last = model.predict(np.expand_dims(state_last, axis=0))
                    q_values_next = model.predict(np.expand_dims(state_next, axis=0))
                    q_values_last[0, action] = reward + discount * np.max(q_values_next)


                    target = np.concatenate((target, q_values_last), axis=0)
                    features = np.concatenate((features, np.expand_dims(state_last, axis=0)))

                model.fit(features, target.astype(float), epochs=1)

            model.save(model_name)
    return 'done'

@app.route("/debug", methods=["POST"])
def debug():
    # endpoint for passing arbitrary data and inserting breakpoints to the python code for debugging
    if flask.request.method == "POST":
        if flask.request.form:
            data = flask.request.form

            data = np.asarray([value for value in data.to_dict().values()])

    return 'done'
vo

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    load_model()
    app.run()
