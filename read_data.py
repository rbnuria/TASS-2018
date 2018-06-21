import csv
import numpy as np
from nltk import word_tokenize, pos_tag, ne_chunk


def readData(url):
	with open(url) as tsvfile:
		reader = csv.DictReader(tsvfile, dialect='excel-tab')

		array_data = []
		array_labels = []

		for row in reader:
			data = row['HEADLINE']
			label = row['TAG']

			vector_sentence = word_tokenize(data)
			while(len(vector_sentence) < 50):
				vector_sentence.append("-")


			array_data.append(vector_sentence)



			if label=='SAFE':
				array_labels.append(1)
			elif label=='UNSAFE':
				array_labels.append(0)


		array_data = np.array(array_data)
		array_labels = np.array(array_labels)


	return (array_data, array_labels)


def readEmbeddings(url):
	embedding = []
	
	with open(url, 'r') as vecfile:
		for line in vecfile:
			split_line = line.split()
			split_line = split_line[:-1]
			embedding.append(split_line)
	
	#Eliminamos primera fila
	embedding = embedding[1:]
	return embedding

