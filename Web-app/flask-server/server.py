from flask import *
from keras import models
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, LSTM, Input
from tensorflow.keras.layers import Input
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import pickle
import os

TextChat = []


def clean_text(txt):
    txt = txt.lower()
    txt = re.sub(r"i'm", "i am", txt)
    txt = re.sub(r"he's", "he is", txt)
    txt = re.sub(r"she's", "she is", txt)
    txt = re.sub(r"that's", "that is", txt)
    txt = re.sub(r"what's", "what is", txt)
    txt = re.sub(r"where's", "where is", txt)
    txt = re.sub(r"\'ll", " will", txt)
    txt = re.sub(r"\'ve", " have", txt)
    txt = re.sub(r"\'re", " are", txt)
    txt = re.sub(r"\'d", " would", txt)
    txt = re.sub(r"won't", "will not", txt)
    txt = re.sub(r"can't", "can not", txt)
    txt = re.sub(r"[^\w\s]", "", txt)
    return txt


dic1_file = open('vocab', 'rb')
dic2_file = open('inv_vocab', 'rb')
vocab = pickle.load(dic1_file)
inv_vocab = pickle.load(dic2_file)


enc_model = models.load_model('enc_model.h5')
dec_model = models.load_model('dec_model.h5')

prepro1 = ""
VOCAB_SIZE = len(vocab)
dense = Dense(VOCAB_SIZE, activation='softmax')
# while prepro1 != 'q':
#     prepro1 = input("you : ")
#     ## prepro1 = "Hello"

#     prepro1 = clean_text(prepro1)
#     ## prepro1 = "hello"

#     prepro = [prepro1]
#     ## prepro1 = ["hello"]

#     txt = []
#     for x in prepro:
#         # x = "hello"
#         lst = []
#         for y in x.split():
#             ## y = "hello"
#             try:
#                 lst.append(vocab[y])
#                 ## vocab['hello'] = 454
#             except:
#                 lst.append(vocab['<OUT>'])
#         txt.append(lst)

#     ## txt = [[454]]
#     txt = pad_sequences(txt, 20, padding='post')

#     # txt = [[454,0,0,0,.........20]]

#     stat = enc_model.predict(txt)

#     empty_target_seq = np.zeros((1, 1))
#     ##   empty_target_seq = [0]

#     empty_target_seq[0, 0] = vocab['<SOS>']
#     ##    empty_target_seq = [255]

#     stop_condition = False
#     decoded_translation = ''

#     while not stop_condition:

#         dec_outputs, h, c = dec_model.predict([empty_target_seq] + stat)
#         decoder_concat_input = dense(dec_outputs)
#         # decoder_concat_input = [0.1, 0.2, .4, .0, ...............]

#         sampled_word_index = np.argmax(decoder_concat_input[0, -1, :])
#         ## sampled_word_index = [2]

#         sampled_word = inv_vocab[sampled_word_index] + ' '

#         ## inv_vocab[2] = 'hi'
#         ## sampled_word = 'hi '

#         if sampled_word != '<EOS> ':
#             decoded_translation += sampled_word

#         if sampled_word == '<EOS> ' or len(decoded_translation.split()) > 20:
#             stop_condition = True

#         empty_target_seq = np.zeros((1, 1))
#         empty_target_seq[0, 0] = sampled_word_index
#         # <SOS> - > hi
#         # hi --> <EOS>
#         stat = [h, c]

# print("Kat : ", decoded_translation)


app = Flask(__name__)

picFolder = os.path.join('static', 'images')

app.config['UPLOAD_FOLDER'] = picFolder
image1 = os.path.join(app.config['UPLOAD_FOLDER'], 'image1.jpg')
image2 = os.path.join(app.config['UPLOAD_FOLDER'], 'image2.jpg')


@app.route("/messageapi", methods=['POST', 'GET'])
def messageapi():
    prepro1 = request.form['inputtext']
    TextChat.append(prepro1)
    ## prepro1 = "Hello"

    prepro1 = clean_text(prepro1)
    ## prepro1 = "hello"

    prepro1 = clean_text(prepro1)
    ## prepro1 = "hello"

    prepro = [prepro1]
    ## prepro1 = ["hello"]

    txt = []
    for x in prepro:
        # x = "hello"
        lst = []
        for y in x.split():
            ## y = "hello"
            try:
                lst.append(vocab[y])
                ## vocab['hello'] = 454
            except:
                lst.append(vocab['<OUT>'])
        txt.append(lst)

    ## txt = [[454]]
    txt = pad_sequences(txt, 20, padding='post')

    # txt = [[454,0,0,0,.........20]]

    stat = enc_model.predict(txt)

    empty_target_seq = np.zeros((1, 1))
    ##   empty_target_seq = [0]

    empty_target_seq[0, 0] = vocab['<SOS>']
    ##    empty_target_seq = [255]

    stop_condition = False
    decoded_translation = ''

    while not stop_condition:

        dec_outputs, h, c = dec_model.predict([empty_target_seq] + stat)
        decoder_concat_input = dense(dec_outputs)
        # decoder_concat_input = [0.1, 0.2, .4, .0, ...............]

        sampled_word_index = np.argmax(decoder_concat_input[0, -1, :])
        ## sampled_word_index = [2]

        sampled_word = inv_vocab[sampled_word_index] + ' '

        ## inv_vocab[2] = 'hi'
        ## sampled_word = 'hi '

        if sampled_word != '<EOS> ':
            decoded_translation += sampled_word

        if sampled_word == '<EOS> ' or len(decoded_translation.split()) > 20:
            stop_condition = True

        empty_target_seq = np.zeros((1, 1))
        empty_target_seq[0, 0] = sampled_word_index
        # <SOS> - > hi
        # hi --> <EOS>
        stat = [h, c]
    TextChat.append(decoded_translation)
    return render_template('index.html', img1=image1, img2=image2, lst=TextChat, length=len(TextChat))


@app.route("/", methods=["GET"])
def index():
    return render_template('index.html', img1=image1, img2=image2, lst=TextChat, length=len(TextChat))


if __name__ == "__main__":
    app.run(debug=True)
