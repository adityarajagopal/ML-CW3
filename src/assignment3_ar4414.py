import data_ops
import learning
import time
import numpy
from sklearn import preprocessing
import matplotlib.pyplot as plt


def q3a(TrainMat, TestMat):
	#training
	UserInd = data_ops.sort_col(TrainMat, 0)
	MovieInd = data_ops.sort_col(TrainMat, 1)
	MoviePredictor = data_ops.mean_by_col(MovieInd, 1, 2, 9066)
	UserPredictor = data_ops.mean_by_col(UserInd, 0, 2, 671)
	#testing
	MovieError =  learning.square_error_a(MoviePredictor, TestMat, 1, 2)
	UserError =  learning.square_error_a(UserPredictor, TestMat, 0, 2)
	print MovieError
	print UserError

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
	#Low = 2
	#High = 4
	#(CVErr, LambdaMat) = learning.cross_val_legendre(TrainMat, FeatureMat, EndPoints, Low, High)
	#BestModelPerUser = numpy.argmin(CVErr, axis=0)
	#BestModel = numpy.bincount(BestModelPerUser).argmax()
	#
	#Z = data_ops.legendre(FeatureMat, BestModel+Low)
	#print Z.shape
	#print "Legendre best degree: ", BestModel + Low
	#(U, MeanMat) = learning.learn(EndPoints, TrainMat, Z, LambdaMat[:,BestModel])

	#Err_Test = learning.square_error_b(Z, U, TestMat, MeanMat)
	#Err_Train = learning.square_error_b(Z, U, TrainMat, MeanMat)
	#print "Legendre Test Error: ", Err_Test
	#print "Legendre Training Error: ", Err_Train
	
	#PCA transformation
	Low = 2
	High = 4
	Z = data_ops.pca(FeatureMat, 3)
	LambdaMat = learning.cross_val_lambda(TrainMat, Z, EndPoints)
	(U, MeanMat) = learning.learn(EndPoints, TrainMat, Z, LambdaMat)
	print "PCA + Poly Training Error: ", learning.square_error_b(Z, U, TrainMat, MeanMat)
	print "PCA + Poly Test Error: ", learning.square_error_b(Z, U, TestMat, MeanMat)

	
	#(CVErr, LambdaMat) = learning.cross_val_poly(TrainMat, Z, EndPoints, Low, High)
	#BestModelPerUser = numpy.argmin(CVErr, axis=0)
	#BestModel = numpy.bincount(BestModelPerUser).argmax()
	#print "PCA + Poly best degree: ", BestModel + Low
	#Poly = preprocessing.PolynomialFeatures(BestModel+1)
	#Z = Poly.fit_transform(Z)
	#(U, MeanMat) = learning.learn(EndPoints, TrainMat, Z, LambdaMat)
	#print "PCA + Poly Training Error: ", learning.square_error_b(Z, U, TrainMat, MeanMat)
	#print "PCA + Poly Test Error: ", learning.square_error_b(Z, U, TestMat, MeanMat)

def q3d(TrainMat, TestMat, NumMovies):
	BestLambda, BestK = learning.n_fold_cross_val(TrainMat, [0.005,0.01,0.05,0.1,1], 0.001, [4,7,10], 1000000, NumMovies, 3889, TestMat)			
	print (BestLambda, BestK)
	Iter = 50000000
	Alpha = 0.0001
	(X, Theta, IterList, TestErr, TrainErr) = learning.collaborative_filter(TrainMat, BestK, BestLambda, Alpha, Iter, NumMovies, TestMat, True, 0) 
	Ratings = numpy.dot(Theta.T, X) 
	print "K: ", BestK, "Lambda: ", BestLambda, "Alpha: ", Alpha, "Iter: ", Iter
	print learning.square_error_collab(Ratings, TrainMat)	
	print learning.square_error_collab(Ratings, TestMat)	
	plt.figure()
	plt.title("Error vs. Iterations")
	plt.xlabel("Iterations")
	plt.ylabel("Error")
	plt.scatter(IterList, TrainErr, c='r', s=1, label="Training Error")
	plt.scatter(IterList, TestErr, c='b', s=1, label = "Test Error")
	plt.legend()
	plt.ylim((0.3,2))
	plt.show()


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
	#q3d(TrainMat, TestMat, FeatureMat.shape[0])


if __name__ == '__main__':
	main()
