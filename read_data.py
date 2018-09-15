import csv
import numpy as np
from nltk import word_tokenize, pos_tag, ne_chunk


def readData(url, max_length):
	with open(url) as tsvfile:
		reader = csv.DictReader(tsvfile, dialect='excel-tab')

		array_data = []
		array_labels = []

		for row in reader:
			data = row['HEADLINE']
			label = row['TAG']

			vector_sentence = word_tokenize(data)
			if(len(vector_sentence)>max_length):
				vector_sentence = vector_sentence[:max_length]
			while(len(vector_sentence) < max_length):
				vector_sentence.append("-")
		

			array_data.append(vector_sentence)


			if label=='SAFE':
				array_labels.append(1)
			elif label=='UNSAFE':
				array_labels.append(0)




		array_data = np.array(array_data)
		array_labels = np.array(array_labels)


	return (array_data, array_labels)


def readDataTest(url, max_length):
	with open(url) as tsvfile:
		reader = csv.DictReader(tsvfile, dialect='excel-tab')

		array_data = []
		array_labels = []
		array_id = []

		for row in reader:
			id_ = row['ID']
			data = row['HEADLINE']

			vector_sentence = word_tokenize(data)
			if(len(vector_sentence)>max_length):
				vector_sentence = vector_sentence[:max_length]
			while(len(vector_sentence) < max_length):
				vector_sentence.append("-")
			
			

			array_data.append(vector_sentence)
			array_id.append(id_)



		array_data = np.array(array_data)
		array_id = np.array(array_id)


	return (array_data, array_id)


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

