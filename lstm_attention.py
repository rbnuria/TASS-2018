import numpy as np
#Fijamos semilla
np.random.seed(666)
from tensorflow import set_random_seed
set_random_seed(2)

from read_data import readData, readEmbeddings
from general import prepareData
from keras.layers import Dense, Dropout, LSTM, Embedding, Bidirectional, Flatten, Activation, RepeatVector, Permute, merge
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
data_train, label_train = readData('tass_2018_task_4_subtask1_train_dev/SANSE_train-1.tsv',input_size)



#a=[[t for t in data if t!='-'] for data in data_train]
#b=[len(t) for t in a]
#print(np.max(b))
#print(np.mean(b))



print(data_train.shape)
print(label_train.shape)

print("Leyendo datos de desarrollo...")
data_dev, label_dev = readData('tass_2018_task_4_subtask1_train_dev/SANSE_dev-1.tsv',input_size)

print(data_dev.shape)
print(label_dev.shape)

print("Leyendo los word embeddings...")
embeddings = KeyedVectors.load_word2vec_format('SBW-vectors-300-min5.bin', binary=True)


print("Transformamos las frases con los embeddings...")
data_train_idx, data_dev_idx, matrix_embeddings, vocab = prepareData(data_train, data_dev, embeddings)




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

#Primera convolución
x = LSTM(units = 256, return_sequences=True)(embedded_sequence)
#x = Bidirectional(LSTM(units = 256, return_sequences=True))(embedded_sequence)
#x = Dropout(0.025)(x)

attention = Dense(1, activation="tanh")(x)
attention = Flatten()(attention)
attention = Activation("softmax")(attention)
attention = RepeatVector(256)(attention)
attention = Permute([2,1])(attention)

x = merge([x, attention], mode="mul")
x = Flatten()(x)
x = Dense(128, activation = "relu", kernel_initializer=glorot_uniform(seed=2), activity_regularizer=regularizers.l2(0.0001))(x)
x = Dropout(0.35)(x)
x = Dense(64, activation = "relu", kernel_initializer=glorot_uniform(seed=2), activity_regularizer=regularizers.l2(0.001))(x)
x = Dropout(0.55)(x)
x = Dense(32, activation = "relu", kernel_initializer=glorot_uniform(seed=2), activity_regularizer=regularizers.l2(0.01))(x)
x = Dropout(0.5)(x)

#Una probabilidad por etiqueta
preds = Dense(2, activation = "softmax")(x)

model = Model(sequence_input, preds)

model.summary()

model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])

earlyStopping = EarlyStopping('loss', patience=5, mode='min')

modelo = model.fit(x = data_train_idx, y = to_categorical(label_train,2), batch_size = 25, epochs = 40, validation_data=(data_dev_idx, to_categorical(label_dev,2)), shuffle = False, callbacks=[earlyStopping])


loss, acc = model.evaluate(x=data_dev_idx, y=to_categorical(label_dev,2), batch_size=25)
print(loss)
print(acc)

y_pred = model.predict(data_dev_idx, batch_size=25)
#EMC: Esto debería estar modularizado
label_dev_tag = ["UNSAFE" if dev==0 else "SAFE" for dev in label_dev]
y_pred_tag = ["UNSAFE" if pred==0 else "SAFE" for pred in np.argmax(y_pred,axis=1)]
dev_pred_tags = zip(label_dev_tag, y_pred_tag,[" ".join(data).strip(" -") for data in data_dev])
with(open("pred_dev_output.tsv", 'w')) as f_pred_out:
    f_pred_out.write("REAL\tPRED\n")
    s_buff = "\n".join(["\t".join(list(label_pair)) for label_pair in dev_pred_tags])
    f_pred_out.write(s_buff)