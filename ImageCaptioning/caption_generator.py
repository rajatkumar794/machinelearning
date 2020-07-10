import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras import Sequential
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_desc(model_sq, tokenizer, photo, max_length):
    in_text = 'startsq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = tf.keras.preprocessing.sequence.pad_sequences([sequence], maxlen=max_length)
        yhat = model_sq.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = word_for_id(yhat, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endsq':
            break
    return in_text

def extract_features():
    vgg_model = VGG16()
    model = Sequential()
    for layer in vgg_model.layers[:-1]:  # this is where I changed your code
        model.add(layer)
    for layer in model.layers:
        layer.trainable = False

    #model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    image = load_img('test2.jpg', target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = model.predict(image, verbose=0)
    feature = np.reshape(feature, (1, 4096))
    return feature

with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pkl.load(handle)

model_sq = tf.keras.models.load_model('model_10.h5')
feature=extract_features()
caption=generate_desc(model_sq, tokenizer, feature, 34)
caption=caption.split(' ')
caption=' '.join(caption[1:-1])
print(caption)
x=plt.imread('test2.jpg')
plt.imshow(x)
plt.show()

