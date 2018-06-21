import numpy as np

def prepareData(train, dev, embeddings):	
	#Almacenamiento de palabras en nuestro vocabulario
	vocabulary = {}
	vocabulary["PADDING"] = len(vocabulary)
	vocabulary["UNKOWN"] = len(vocabulary)

	train_idx = []
	dev_idx = []

	#Matriz de embeddings del vocabulario
	embeddings_matrix = []
	embeddings_matrix.append(np.zeros(300))
	embeddings_matrix.append(np.random.uniform(-0.25, 0.25, 300))

	for sentence in train:
		wordIndices = []
		for word in sentence:
			#Si tenemos embedding para la palabra
			if word in embeddings:
				#Si ya estaba en nuestro vocabulario
				if word in vocabulary:
					wordIndices.append(vocabulary[word])
					#Ya estará en embeddings
				else:
					vocabulary[word] = len(vocabulary)
					wordIndices.append(vocabulary[word])
					embeddings_matrix.append(embeddings[word])

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
			if word in embeddings:
				#Si ya estaba en nuestro vocabulario
				if word in vocabulary:
					wordIndices.append(vocabulary[word])
					#Ya estará en embeddings
				else:
					vocabulary[word] = len(vocabulary)
					wordIndices.append(vocabulary[word])
					embeddings_matrix.append(embeddings[word])

			else:
				#Padding
				if word == "-":
					wordIndices.append(vocabulary["PADDING"])
				#Desconocida
				else:
					wordIndices.append(vocabulary["UNKOWN"])

		dev_idx.append(np.array(wordIndices))

	return (train_idx, dev_idx, embeddings_matrix, vocabulary)
