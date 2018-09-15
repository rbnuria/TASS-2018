import numpy as np

def prepareData(train, dev, embeddings):	
	#Almacenamiento de palabras en nuestro vocabulario -> Añadimos al vocabulario
	#aquellas palabras que estén en el subconjunto de los embeddings seleccionados 
	#(200000) + 2 de padding y unkown

	vocabulary = {}
	vocabulary["PADDING"] = len(vocabulary)
	vocabulary["UNKOWN"] = len(vocabulary)

	#Matriz de embeddings del vocabulario
	embeddings_matrix = []
	embeddings_matrix.append(np.zeros(300))
	embeddings_matrix.append(np.random.uniform(-0.25, 0.25, 300))

	for word in embeddings.wv.vocab:
		vocabulary[word] = len(vocabulary)
		#Al mismo tiempo creamos matrix de embeddings
		embeddings_matrix.append(embeddings[word])


	train_idx = []
	dev_idx = []

	for sentence in train:
		wordIndices = []
		for word in sentence:
			#Si la palabra está en el vocabulario, asignamos su índice en él
			if word in vocabulary:
				wordIndices.append(vocabulary[word])
			else:
				#Padding
				if word == "-":
					wordIndices.append(vocabulary["PADDING"])
				#Desconocida
				else:
					wordIndices.append(vocabulary["UNKOWN"])

		train_idx.append(np.array(wordIndices))

	for sentence in dev:
		wordIndices = []
		for word in sentence:
			#Si tenemos embedding para la palabra
			if word in vocabulary:
				wordIndices.append(vocabulary[word])
			else:
				#Padding
				if word == "-":
					wordIndices.append(vocabulary["PADDING"])
				#Desconocida
				else:
					wordIndices.append(vocabulary["UNKOWN"])

		dev_idx.append(np.array(wordIndices))

	return (train_idx, dev_idx, embeddings_matrix, vocabulary)


def prepareDataTest(data_test, vocabulary):

	data_test_idx = []

	for sentence in data_test:
		wordIndices = []
		for word in sentence:
			#Si tenemos embedding para la palabra
			if word in vocabulary:
				wordIndices.append(vocabulary[word])
			else:
				#Padding
				if word == "-":
					wordIndices.append(vocabulary["PADDING"])
				#Desconocida
				else:
					wordIndices.append(vocabulary["UNKOWN"])

		data_test_idx.append(np.array(wordIndices))

	return np.array(data_test_idx)


def writeOutput(y_pred, id_, fichero):
	y_pred_tag = ["UNSAFE" if pred==0 else "SAFE" for pred in np.argmax(y_pred,axis=1)]
	output_data = zip(id_, y_pred_tag)

	with(open(fichero, 'w')) as f_test_out:
		s_buff = "\n".join(["\t".join(list(label_pair)) for label_pair in output_data])
		f_test_out.write(s_buff)

