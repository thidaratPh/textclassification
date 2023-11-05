
from flask import Flask, request, jsonify
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import keras
import numpy as np

lr = pickle.load(open('tokenizer_p.pkl','rb'))

app = Flask(__name__)
# Routes
@app.route("/")
def index():
    return jsonify({"message": "Hello"})





def predict_result(text):
    result = {0: "REAL REVIEW", 1: "GIBBERISH"}

    vocab_size = len(lr.word_index) + 1
    model = keras.models.Sequential([
    keras.layers.Embedding(vocab_size, 64),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dropout(rate=0.25),
    keras.layers.Dense(2, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["acc"])

    # Convert the text to a numerical sequence
    integer_sequence = lr.texts_to_sequences([text])

    # Make a prediction using the pre-trained model
    predictions = model.predict(integer_sequence)

    # Get the predicted class
    predicted_class = np.argmax(predictions[0])

    # Print the prediction result to the console
    return(result[predicted_class])

@app.route("/api/predict", methods=['POST'])
def read_str():

    try:
        data=request.get_json()['itme_str']

        result=predict_result(data)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'erro':str(e)})
        




if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)

# import tensorflow
# print(tensorflow.__version__)
