import numpy
import math

def square_error_a(TrainMatrix, TestMatrix, DataCol, RatingCol):
	AbsError = [TrainMatrix[int(Row[DataCol])-1, 1] - Row[RatingCol] for Row in TestMatrix]
	SqError = numpy.square(AbsError)
	CumSqError = numpy.sum(SqError)
	return CumSqError/TestMatrix.shape[0]

def square_error_b(FeatMat, U, TestMat):
	R = numpy.dot(FeatMat, U)
	AbsError = [R[int(Row[1]-1), int(Row[0]-1)] - Row[2] for Row in TestMat] 
	SqError = numpy.square(AbsError)
	CumSqError = numpy.sum(SqError)
	return CumSqError/TestMat.shape[0]
	
def ridge_regression(Z, Y, Lambda):
	Zt = numpy.transpose(Z, (1,0))
	ZtZ = numpy.dot(Zt,Z)
	LambdaI = numpy.zeros((ZtZ.shape[0], ZtZ.shape[1]))
	numpy.fill_diagonal(LambdaI, Lambda)
	Tmp = ZtZ + LambdaI
	Tmp1 = numpy.linalg.inv(Tmp)
	Tmp2 = numpy.dot(Tmp1, Zt)
	return numpy.dot(Tmp2, Y)

def n_fold_cross_val(TrainMat, FeatMat, EndPoints):
	LambdaUserMat = numpy.zeros((1,1))
	#SortedTrain = data_ops.sort_col(TrainMat, 0)
	#EndPoints = data_ops.extract_endpoints(SortedTrain, 0)
	LambdaList = []

	for End in EndPoints: 
		Start = 0
		BestValError = 10000000000000
		BestLambda = 0
		
		Tmp = TrainMat[Start:End+1, 1:3]
		Start = End+1
		Z = numpy.zeros((1,FeatMat.shape[1]))
		#Y = data_ops.normalise(Tmp[:,1])
		Y = numpy.array([Tmp[:,1]])
		Y = Y.T	
		for Index in Tmp[:,0]:
			Z = numpy.append(Z, [FeatMat[int(Index-1),:]], axis=0)
		Z = numpy.delete(Z, (0), axis=0)
		#Z = data_ops.normalise(Z)
		
		for Lambda in numpy.arange(0.05,10.05,0.5):
			TotalValError = 0
			#Lambda = math.pow(10, i)
			
			for Index in xrange(0,Z.shape[0]):
				Z_train = numpy.delete(Z, (Index), axis=0)
				Y_train = numpy.delete(Y, (Index), axis=0)
				Ui = ridge_regression(Z_train, Y_train, Lambda)
				#get validation error 
				ZVal = numpy.array([Z[Index,:]])
				RVal = numpy.dot(ZVal, Ui)
				TotalValError += numpy.square(RVal - Y[Index,:])
			ValError = TotalValError / Z.shape[0]
			if ValError < BestValError : 
				BestValError = ValError
				BestLambda = Lambda
		LambdaUserMat = numpy.append(LambdaUserMat, [[BestLambda]], axis=0)
		print End
		print LambdaUserMat
	LambdaUserMat = numpy.delete(LambdaUserMat, (0), axis=0)
	return LambdaUserMat
	



		
			











