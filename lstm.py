import numpy as np
#Fijamos semilla
np.random.seed(666)
from tensorflow import set_random_seed
set_random_seed(2)

from read_data import readData, readEmbeddings
from general import prepareData
from keras.layers import Dense, Dropout, LSTM, Embedding
from keras.models import Model
from keras.layers.core import Activation
from keras.layers import Embedding
from keras.layers import Input
from keras.layers.core import Dropout
from keras.layers.core import Dense
from gensim.models.keyedvectors import KeyedVectors
from keras import optimizers
from keras.utils import to_categorical
from keras import regularizers



#Lectura de los datos

print("Leyendo datos de entrenamiento...")
data_train, label_train = readData('tass_2018_task_4_subtask1_train_dev/SANSE_train-1.tsv')

print(data_train.shape)
print(label_train.shape)

print("Leyendo datos de desarrollo...")
data_dev, label_dev = readData('tass_2018_task_4_subtask1_train_dev/SANSE_dev-1.tsv')

print(data_dev.shape)
print(label_dev.shape)

print("Leyendo los word embeddings...")
embeddings = KeyedVectors.load_word2vec_format('SBW-vectors-300-min5.bin', binary=True)


print("Transformamos las frases con los embeddings...")
data_train_idx, data_dev_idx, matrix_embeddings, vocab = prepareData(data_train, data_dev, embeddings)

data_train_idx = np.array(data_train_idx)
data_dev_idx = np.array(data_dev_idx)
matrix_embeddings = np.array(matrix_embeddings)



######################Configuración parámetros
input_size = 50

sequence_input = Input(shape = (input_size, ), dtype = 'float64')
embedding_layer = Embedding(matrix_embeddings.shape[0], matrix_embeddings.shape[1], weights=[matrix_embeddings],trainable=False, input_length = input_size) #Trainable false
embedded_sequence = embedding_layer(sequence_input)

#Primera convolución
x = LSTM(units = 128)(embedded_sequence)
x = Dropout(0.5)(x)
x = Dense(100, activation = "tanh", activity_regularizer=regularizers.l2(0.05))(x)
x = Dropout(0.5)(x)
x = Dense(75, activation = "tanh", activity_regularizer=regularizers.l2(0.05))(x)
x = Dropout(0.5)(x)
x = Dense(50, activation = "tanh", activity_regularizer=regularizers.l2(0.05))(x)
x = Dropout(0.5)(x)

#Una probabilidad por etiqueta
preds = Dense(2, activation = "softmax")(x)

model = Model(sequence_input, preds)

model.summary()

model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])

modelo = model.fit(x = data_train_idx, y = to_categorical(label_train,2), batch_size = 64, epochs = 20, validation_data=(data_dev_idx, to_categorical(label_dev,2)), shuffle = False)


loss, acc = model.evaluate(x=data_dev_idx, y=to_categorical(label_dev,2), batch_size=64)
print(loss)
print(acc)

y_pred = model.predict(data_train_idx)

