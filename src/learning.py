import numpy
import math
import data_ops
from sklearn import preprocessing

def square_error_a(TrainMatrix, TestMatrix, DataCol, RatingCol):
	AbsError = [TrainMatrix[int(Row[DataCol])-1, 1] - Row[RatingCol] for Row in TestMatrix]
	SqError = numpy.square(AbsError)
	CumSqError = numpy.sum(SqError)
	return CumSqError/TestMatrix.shape[0]

def square_error_b(FeatMat, U, TestMat, MeanMat):
	FeatMat = preprocessing.scale(FeatMat)
	R = numpy.dot(FeatMat, U)
	R = R + MeanMat
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
	LambdaList = []
	Start = 0

	for End in EndPoints: 
		Tmp = TrainMat[Start:End+1, 1:3]
		Start = End+1
		Z = numpy.zeros((1,FeatMat.shape[1]))
		Y = numpy.array([Tmp[:,1]])
		Y = Y.T	
		Y = Y - numpy.mean(Y, axis=0)

		for Index in Tmp[:,0]:
			Z = numpy.append(Z, [FeatMat[int(Index-1),:]], axis=0)
		Z = numpy.delete(Z, (0), axis=0)
		Z = preprocessing.scale(Z)
		
		#(BestLambda, BestValError) = empirical_cross_val(Z, Y)
		BestLambda = analytical_cross_val(Z,Y)
		LambdaUserMat = numpy.append(LambdaUserMat, [[BestLambda]], axis=0)
	LambdaUserMat = numpy.delete(LambdaUserMat, (0), axis=0)
	return LambdaUserMat

def analytical_cross_val(Z, Y):
	BestLambda = 0
	BestValError = 1000000
	MinLambda = 0.0
	MaxLambda = 3000.0
	Steps = 100
	StepSize = (MaxLambda - MinLambda)/Steps
	for Lambda in numpy.arange(MinLambda+StepSize, MaxLambda+StepSize, StepSize):
		H = H_cal(Z, Lambda)
		Hnn = numpy.array([numpy.diag(H)]).T
		Y_hat = numpy.dot(H, Y)
		N = Z.shape[0]
		Tmp1 = Y_hat - Y
		Tmp2 = 1 - Hnn
		Tmp3 = numpy.square(Tmp1 / Tmp2)
		Tmp4 = numpy.sum(Tmp3)
		ValError = Tmp4 / N
		if ValError < BestValError:
			BestValError = ValError 
			BestLambda = Lambda
	return BestLambda

def H_cal(Z, Lambda):
	Zt = numpy.transpose(Z, (1,0))
	ZtZ = numpy.dot(Zt,Z)
	LambdaI = numpy.zeros((ZtZ.shape[0], ZtZ.shape[1]))
	numpy.fill_diagonal(LambdaI, Lambda)
	Tmp = ZtZ + LambdaI
	Tmp1 = numpy.linalg.inv(Tmp)
	Tmp2 = numpy.dot(Tmp1, Zt)
	return numpy.dot(Z, Tmp2)


def empirical_cross_val(Z, Y):
	BestValError = 1000000
	BestLambda = 0
	
	for Lambda in numpy.arange(0.05,10.05,0.05):
		TotalValError = 0
		
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
	
	return (BestLambda, BestValError)
	


	



		
			











