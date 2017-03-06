import data_ops
import learning
import time
import numpy
from sklearn import preprocessing


def q3a(TrainMat, TestMat):
	#training
	UserInd = data_ops.sort_col(TrainMat, 1)
	MovieInd = data_ops.sort_col(TrainMat, 0)
	MoviePredictor = data_ops.mean_by_col(UserInd, 1, 2, 9066)
	UserPredictor = data_ops.mean_by_col(MovieInd, 0, 2, 671)

	#testing
	MovieError =  learning.square_error_a(MoviePredictor, TestMat, 1, 2)
	UserError =  learning.square_error_a(UserPredictor, TestMat, 0, 2)

#def q3b(TrainMat, TestMat, FeatMat):
#	SortedTrain = data_ops.sort_col(TrainMat, 0)
#	EndPoints = data_ops.extract_endpoints(SortedTrain, 0)
#	Start = 0
#	Count = 0
#	U = numpy.zeros((FeatMat.shape[1], 1))
#	
#	for End in EndPoints: 
#		Tmp = TrainMat[Start:End+1, 1:3]
#		Start = End+1
#		Z = numpy.zeros((1,FeatMat.shape[1]))
#		#Y = data_ops.normalise(Tmp[:,1])
#		Y = numpy.array([Tmp[:,1]])
#		Y = Y.T	
#		for Index in Tmp[:,0]:
#			Z = numpy.append(Z, [FeatMat[int(Index-1),:]], axis=0)
#		Z = numpy.delete(Z, (0), axis=0)
#		#Z = data_ops.normalise(Z)
#		Ui = learning.ridge_regression(Z, Y, 3.15)
#		U = numpy.append(U, Ui, axis=1)
#	U = numpy.delete(U, (0), axis=1)
#
#	Err = learning.square_error_b(FeatMat, U, TestMat)
#	print Err
		
def q3b(TrainMat, TestMat, FeatMat):
	SortedTrain = data_ops.sort_col(TrainMat, 0)
	EndPoints = data_ops.extract_endpoints(SortedTrain, 0)
	Start = 0
	Index1 = 0
	U = numpy.zeros((FeatMat.shape[1], 1))
	MeanMat = numpy.zeros((1,1))

	LUMat = learning.n_fold_cross_val(TrainMat, FeatMat, EndPoints)
	print LUMat
	
	for End in EndPoints: 
		Tmp = TrainMat[Start:End+1, 1:3]
		Start = End+1
		Z = numpy.zeros((1,FeatMat.shape[1]))
		#Y = data_ops.normalise(Tmp[:,1])
		Y = numpy.array([Tmp[:,1]])
		Y = Y.T	
		#center Y
		Mean = numpy.mean(Y, axis=0)
		MeanMat = numpy.append(MeanMat, [Mean], axis=1)
		Y = Y - Mean 
		
		for Index in Tmp[:,0]:
			Z = numpy.append(Z, [FeatMat[int(Index-1),:]], axis=0)
		Z = numpy.delete(Z, (0), axis=0)
		Z = preprocessing.scale(Z)	

		Ui = learning.ridge_regression(Z, Y, LUMat[Index1])
		U = numpy.append(U, Ui, axis=1)
		Index1 += 1
	U = numpy.delete(U, (0), axis=1)
	MeanMat = numpy.delete(MeanMat, (0), axis=1)

	Err_Test = learning.square_error_b(FeatMat, U, TestMat, MeanMat)
	Err_Train = learning.square_error_b(FeatMat, U, TrainMat, MeanMat)
	print Err_Test
	print Err_Train

def main():
	TrainingData = '../movie-data/ratings-train.csv'
	TrainMat = data_ops.reader(TrainingData)
	
	TestData = '../movie-data/ratings-test.csv'
	TestMat = data_ops.reader(TestData)
	
	FeatureData = '../movie-data/movie-features.csv'
	FeatureMat = data_ops.reader(FeatureData)
	FeatureMat = numpy.delete(FeatureMat, (0), axis=1)

	#q3a(TrainMat, TestMat)
	q3b(TrainMat, TestMat, FeatureMat)


if __name__ == '__main__':
	main()
