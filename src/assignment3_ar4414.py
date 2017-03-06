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

def q3b(TrainMat, TestMat, FeatMat):
	SortedTrain = data_ops.sort_col(TrainMat, 0)
	EndPoints = data_ops.extract_endpoints(SortedTrain, 0)

	LUMat = learning.cross_val_lambda(TrainMat, FeatMat, EndPoints)
	print LUMat
	
	(U, MeanMat) = learning.learn(EndPoints, TrainMat, FeatMat, LUMat)

	Err_Test = learning.square_error_b(FeatMat, U, TestMat, MeanMat)
	Err_Train = learning.square_error_b(FeatMat, U, TrainMat, MeanMat)
	print Err_Test
	print Err_Train

def q3c(TrainMat, TestMat, FeatureMat):	
	SortedTrain = data_ops.sort_col(TrainMat, 0)
	EndPoints = data_ops.extract_endpoints(SortedTrain, 0)

	#Legendre transformation
	Low = 2
	High = 4
	(CVErr, LambdaMat) = learning.cross_val_legendre(TrainMat, FeatureMat, EndPoints, Low, High)
	BestModelPerUser = numpy.argmin(CVErr, axis=0)
	BestModel = numpy.bincount(BestModelPerUser).argmax()
	
	Z = data_ops.legendre(FeatureMat, BestModel+Low)
	print BestModel + Low
	(U, MeanMat) = learning.learn(EndPoints, TrainMat, Z, LambdaMat[:,BestModel])

	Err_Test = learning.square_error_b(Z, U, TestMat, MeanMat)
	Err_Train = learning.square_error_b(Z, U, TrainMat, MeanMat)
	print Err_Test
	print Err_Train
	
	#PCA transformation
	#Z = data_ops.pca(FeatureMat, 5)

def main():
	TrainingData = '../movie-data/ratings-train.csv'
	TrainMat = data_ops.reader(TrainingData)
	
	TestData = '../movie-data/ratings-test.csv'
	TestMat = data_ops.reader(TestData)
	
	FeatureData = '../movie-data/movie-features.csv'
	FeatureMat = data_ops.reader(FeatureData)
	FeatureMat = numpy.delete(FeatureMat, (0), axis=1)

	#q3a(TrainMat, TestMat)
	#q3b(TrainMat, TestMat, FeatureMat)
	q3c(TrainMat, TestMat, FeatureMat)


if __name__ == '__main__':
	main()
