from skimage import io
from skimage import color
import glob, sys, traceback, os
import numpy as np
from random import shuffle
from skimage.feature import local_binary_pattern
from sklearn.metrics import f1_score, precision_score, recall_score

'''
Essa funcao eh responsavel por carregar as imagens de face de um determinado individuo.
Return: Uma lista com o conjunto de imagens.
'''
def carregarImagens(path):
	image_list = []
	for filename in glob.glob(path + '*.pgm'): 
		im = io.imread(filename)
		image_list.append(im)
	return image_list

	'''
Obter conjuntos de treino e teste
Return: 1) Imagens do conjunto de treino;
		2) Imagens do conjunto de teste.
'''
def processarIndividuo(i):
	LBP = []
	#Faces do individuo i 
	image_list = carregarImagens(str(os.getcwd()) + '/att_faces/s' + str(i) + '/')
	#misturados
	shuffle(image_list)
	#conjunto de treino
	train = image_list[:7] 
	#conjunto de teste
	test = image_list[7:10] 

	return train, test
'''
Extrai LBP de cada imagem do conjunto de treino e obtem um histograma. 
Histograma medio eh obtido e utilizado como descritor de cada face.
'''
def extrairVetorDeCaracteristicasTreino(input_list):
	sum_of_histograms = 0
	for im in input_list:
		#num_points = 24, #radius = 8
		lbp = local_binary_pattern(im, 24, 8, 'uniform')
		(hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 27), range=(0, 26))
		sum_of_histograms += hist
	avg_hist = sum_of_histograms / len(input_list)	
	return avg_hist

def extrairVetorDeCaracteristicasTeste(input):
	#num_points = 24, #radius = 8
	lbp = local_binary_pattern(input, 24, 8, 'uniform')
	(hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 27), range=(0, 26))
	return hist

'''
Funcao para calcular distancia euclidiana no caso n-dimensional.
'''
def euclidianDistance(a, b):
	dist = (a - b)**2
	dist = np.sum(dist)
	dist = np.sqrt(dist)
	return dist


if __name__ == '__main__':
	try:

		lbp_trainset = []

		lbp_testset = [[0 for x in range(3)] for y in range(40)] 

		for i in range(1, 41):
			print "Obtendo conjunto de treino e teste da face " + str(i)
			trainset_i, testset_i = processarIndividuo(i)
			print "Estraindo vetor de caracteristicas dos das faces do conjuntos de treino e teste da face " + str(i)
			lbp_trainset_i = extrairVetorDeCaracteristicasTreino(trainset_i)
			lbp_trainset.append(lbp_trainset_i)

			for j in range(0, 3):
				lbp_testset[i-1][j] = extrairVetorDeCaracteristicasTeste(testset_i[j])

			print "----- // -----"

		print "Computando distancia euclidiana dos vetores do conjunto de teste com os do conjunto de treino" 

		result_list = []
		for lbp_test in lbp_testset:
			for image in lbp_test:
				c=1000000000
				result = 0
				for j in range(0, 40):
					dist = euclidianDistance(image, lbp_trainset[j])
					if (dist < c):
						c = dist
						result = j
				result_list.append(result)

		print "----- // -----"
		print "Resutado da classificacao de texturas: "
		print result_list

		y_test = ['s1', 's1', 's1', 's2', 's2', 's2', 's3', 's3', 's3', 's4', 's4', 's4', 's5', 's5', 's5', 's6', 's6', 's6', 
		's7', 's7', 's7', 's8', 's8', 's8', 's9', 's9', 's9', 's10', 's10', 's10', 's11', 's11', 's11', 's12', 's12', 's12', 
		's13', 's13', 's13', 's14', 's14', 's14', 's15', 's15', 's15', 's16', 's16', 's16', 's17', 's17', 's17', 's18', 's18', 's18', 
		's19', 's19', 's19', 's20', 's20', 's20', 's21', 's21', 's21', 's22', 's22', 's22', 's23', 's23', 's23', 's24', 's24', 's24', 
		's25', 's25', 's25', 's26', 's26', 's26', 's27', 's27', 's27', 's28', 's28', 's28', 's29', 's29', 's29', 's30', 's30', 's30', 
		's31', 's31', 's31', 's32', 's32', 's32', 's33', 's33', 's33', 's34', 's34', 's34', 's35', 's35', 's35', 's36', 's36', 's36', 
		's37', 's37', 's37', 's38', 's38', 's38', 's39', 's39', 's39', 's40', 's40', 's40']
		y_pred = []
		for r in range(0, 120):
			face_r = "s" + str(result_list[r]+1)
			y_pred.append(face_r)

		print "----- // -----"
		print "Predicoes: "
		print y_pred
		print "----- // -----"
		print "Metricas: "
		print "F1_SCORE: " + str(f1_score(y_test, y_pred, average="macro"))
		print "PRECISION: " + str(precision_score(y_test, y_pred, average="macro"))
		print "RECALL: " + str(recall_score(y_test, y_pred, average="macro"))

		sys.exit(0)
	except Exception, e:
		print 'ERROR, UNEXPECTED EXCEPTION'
		print str(e)
		traceback.print_exc()
		sys.exit(1)