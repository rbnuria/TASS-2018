import numpy as np

#Fijamos semilla
np.random.seed(666)
from tensorflow import set_random_seed
set_random_seed(2)

from read_data import readData, readEmbeddings, readDataTest
from general import prepareData, writeOutput, prepareDataTest
from keras.layers import Dense, Dropout, LSTM, Embedding, Bidirectional
from keras.models import Model
from keras.layers.core import Activation
from keras.layers import Embedding
from keras.layers import Input
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.callbacks import EarlyStopping
from keras.initializers import glorot_normal, glorot_uniform
from gensim.models.keyedvectors import KeyedVectors
from keras import optimizers
from keras.utils import to_categorical
from keras import regularizers



#Lectura de los datos
input_size = 20
print("Leyendo datos de entrenamiento...")
data_train, label_train = readData('tass_2018_task_4_subtask2_train_dev/SANSE_train-2.tsv',input_size)


print(data_train.shape)
print(label_train.shape)

print("Leyendo datos de desarrollo...")
data_dev, label_dev = readData('tass_2018_task_4_subtask2_train_dev/SANSE_dev-2.tsv',input_size)

print(data_dev.shape)
print(label_dev.shape)

print("Leyendo datos de test...")
data_test_1, id_test_1 = readDataTest('/Users/nuria/SEPLN/test-s2.tsv',input_size)
#data_test_2, id_test_2 = readDataTest('/Users/nuria/SEPLN/tass_2018_task_4_subtask1_test_l1_l2/test-s1-l2.tsv',input_size)



print("Leyendo los word embeddings...")
embeddings = KeyedVectors.load_word2vec_format('SBW-vectors-300-min5.bin', binary=True)


print("Transformamos las frases con los embeddings...")
data_train_idx, data_dev_idx, matrix_embeddings, vocab = prepareData(data_train, data_dev, embeddings)

data_test_1 = prepareDataTest(data_test_1, vocab)
#data_test_2 = prepareDataTest(data_test_2, vocab)


data_train_idx = np.array(data_train_idx)
data_dev_idx = np.array(data_dev_idx)
matrix_embeddings = np.array(matrix_embeddings)

print(data_train_idx.shape)
print(data_dev_idx.shape)
print(matrix_embeddings.shape)


######################Configuración parámetros
sequence_input = Input(shape = (input_size, ), dtype = 'float64')
embedding_layer = Embedding(matrix_embeddings.shape[0], matrix_embeddings.shape[1], weights=[matrix_embeddings],trainable=False, input_length = input_size) #Trainable false
embedded_sequence = embedding_layer(sequence_input)

x = Bidirectional(LSTM(units = 256))(embedded_sequence)
x = Dense(256, activation = "relu", kernel_initializer=glorot_uniform(seed=2), activity_regularizer=regularizers.l2(0.0001))(x)
x = Dropout(0.35)(x)
x = Dense(128, activation = "relu", kernel_initializer=glorot_uniform(seed=2), activity_regularizer=regularizers.l2(0.001))(x)
x = Dropout(0.35)(x)
x = Dense(32, activation = "relu", kernel_initializer=glorot_uniform(seed=2), activity_regularizer=regularizers.l2(0.01))(x)
x = Dropout(0.5)(x)

preds = Dense(2, activation = "softmax")(x)

model = Model(sequence_input, preds)
model.summary()

model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])

earlyStopping = EarlyStopping('loss', patience=5, mode='min')

modelo = model.fit(x = data_train_idx, y = to_categorical(label_train,2), batch_size = 25, epochs = 40, validation_data=(data_dev_idx, to_categorical(label_dev,2)), shuffle = False, callbacks=[earlyStopping])


loss, acc = model.evaluate(x=data_dev_idx, y=to_categorical(label_dev,2), batch_size=25)
#print(loss)
#print(acc)

y_pred_1 = model.predict(data_test_1, batch_size=25)
#y_pred_2 = model.predict(data_test_2, batch_size=25)

writeOutput(y_pred_1, id_test_1, "bi_s2.txt")
