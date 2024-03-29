import numpy
import math
import data_ops
import random
from sklearn import preprocessing

def square_error_a(TrainMatrix, TestMatrix, DataCol, RatingCol):
	AbsError = [TrainMatrix[int(Row[DataCol])-1, 1] - Row[RatingCol] if TrainMatrix[int(Row[DataCol])-1,1] != 0 else 0 for Row in TestMatrix]
	NonZero = filter(lambda a: a != 0, AbsError)
	SqError = numpy.square(AbsError)
	CumSqError = numpy.sum(SqError)
	return CumSqError/len(NonZero)

def square_error_b(FeatMat, U, TestMat, MeanMat):
	FeatMat = preprocessing.scale(FeatMat)
	R = numpy.dot(FeatMat, U)
	R = R + MeanMat
	AbsError = [R[int(Row[1]-1), int(Row[0]-1)] - Row[2] for Row in TestMat] 
	SqError = numpy.square(AbsError)
	CumSqError = numpy.sum(SqError)
	return CumSqError/TestMat.shape[0]

def square_error_collab(Ratings, TestMat):
	SqErr = [numpy.square(Row[2] - Ratings[int(Row[0]-1), int(Row[1]-1)]) for Row in TestMat]
	return numpy.sum(SqErr) / TestMat.shape[0]		
		
	
def ridge_regression(Z, Y, Lambda):
	Zt = numpy.transpose(Z, (1,0))
	ZtZ = numpy.dot(Zt,Z)
	LambdaI = numpy.zeros((ZtZ.shape[0], ZtZ.shape[1]))
	numpy.fill_diagonal(LambdaI, Lambda)
	Tmp = ZtZ + LambdaI
	Tmp1 = numpy.linalg.inv(Tmp)
	Tmp2 = numpy.dot(Tmp1, Zt)
	return numpy.dot(Tmp2, Y)

def cross_val_lambda(TrainMat, FeatMat, EndPoints):
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
		
		BestLambda = analytical_lambda_cross_val(Z,Y)
		LambdaUserMat = numpy.append(LambdaUserMat, [[BestLambda]], axis=0)
	LambdaUserMat = numpy.delete(LambdaUserMat, (0), axis=0)
	return LambdaUserMat

def analytical_lambda_cross_val(Z, Y):
	BestLambda = 0
	BestValError = 1000000
	MinLambda = 0.0 
	MaxLambda = 10000.0
	Steps = 10
	StepSize = (MaxLambda - MinLambda)/Steps
	for Lambda in numpy.arange(MinLambda+StepSize, MaxLambda+StepSize, StepSize):
		ValError = cv_error(Z, Y, Lambda)
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

def cv_error(Z, Y, Lambda):
	H = H_cal(Z, Lambda)
	Hnn = numpy.array([numpy.diag(H)]).T
	Y_hat = numpy.dot(H, Y)
	N = Z.shape[0]
	Tmp1 = Y_hat - Y
	Tmp2 = 1 - Hnn
	Tmp3 = numpy.square(Tmp1 / Tmp2)
	Tmp4 = numpy.sum(Tmp3)
	return (Tmp4 / N)
	
def cross_val_legendre(TrainMat, FeatMat, EndPoints, LowDeg, Degree):
	CVErrMat = numpy.zeros((Degree-(LowDeg-1), len(EndPoints)))
	LambdaMat = numpy.zeros((len(EndPoints), 1))
	
	for Deg in xrange(LowDeg, Degree+1):
		ZLeg = data_ops.legendre(FeatMat, Deg)
		LUMat = cross_val_lambda(TrainMat, ZLeg, EndPoints)
		LambdaMat = numpy.append(LambdaMat, LUMat, axis=1)
		
		Start = 0
		User = 0
		for End in EndPoints:
			Tmp = TrainMat[Start:End+1, 1:3]
			Start = End+1
			Z = numpy.zeros((1,ZLeg.shape[1]))
			Y = numpy.array([Tmp[:,1]])
			Y = Y.T	
			Y = Y - numpy.mean(Y, axis=0)

			for Index in Tmp[:,0]:
				Z = numpy.append(Z, [ZLeg[int(Index-1),:]], axis=0)
			Z = numpy.delete(Z, (0), axis=0)
			Z = preprocessing.scale(Z)

			CrossValErr = cv_error(Z, Y, LUMat[User])
			CVErrMat[Deg-LowDeg, User] = CrossValErr
			User += 1
	LambdaMat = numpy.delete(LambdaMat, (0), axis=1)	
	return (CVErrMat, LambdaMat)

def cross_val_poly(TrainMat, FeatMat, EndPoints, LowDeg, Degree):
	CVErrMat = numpy.zeros((Degree-(LowDeg-1), len(EndPoints)))
	LambdaMat = numpy.zeros((len(EndPoints), 1))
	
	for Deg in xrange(LowDeg, Degree+1):
		Poly = preprocessing.PolynomialFeatures(Deg, interaction_only=False)
		ZLeg = Poly.fit_transform(FeatMat) 
		print ZLeg.shape
		LUMat = cross_val_lambda(TrainMat, ZLeg, EndPoints)
		LambdaMat = numpy.append(LambdaMat, LUMat, axis=1)
		
		Start = 0
		User = 0
		for End in EndPoints:
			Tmp = TrainMat[Start:End+1, 1:3]
			Start = End+1
			Z = numpy.zeros((1,ZLeg.shape[1]))
			Y = numpy.array([Tmp[:,1]])
			Y = Y.T	
			Y = Y - numpy.mean(Y, axis=0)

			for Index in Tmp[:,0]:
				Z = numpy.append(Z, [ZLeg[int(Index-1),:]], axis=0)
			Z = numpy.delete(Z, (0), axis=0)
			Z = preprocessing.scale(Z)

			CrossValErr = cv_error(Z, Y, LUMat[User])
			CVErrMat[Deg-LowDeg, User] = CrossValErr
			User += 1
	LambdaMat = numpy.delete(LambdaMat, (0), axis=1)	
	return (CVErrMat, LambdaMat)

def learn(EndPoints, TrainMat, FeatMat, LUMat):
	Start = 0
	Index1 = 0
	U = numpy.zeros((FeatMat.shape[1], 1))
	MeanMat = numpy.zeros((1,1))
	
	for End in EndPoints: 
		Tmp = TrainMat[Start:End+1, 1:3]
		Start = End+1
		Z = numpy.zeros((1,FeatMat.shape[1]))
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

		Ui = ridge_regression(Z, Y, LUMat[Index1])
		U = numpy.append(U, Ui, axis=1)
		Index1 += 1
	U = numpy.delete(U, (0), axis=1)
	MeanMat = numpy.delete(MeanMat, (0), axis=1)

	return (U, MeanMat)

def stoch_grad_descent(Theta, X, Iterations, Alpha, Lambda, SortedUser, SortedMovie, UserEndPoints, MovieEndPoints, TrainMat, TestMat, Graph, Model):
	XNew = X
	ThetaNew = Theta
	SampleNum = Iterations/100 
	IterList = [i for i in xrange(0, Iterations+1, SampleNum)]
	TestErr = []
	TrainErr = []
	for i in xrange(0, Iterations+1): 
		# update X's
		for Movie in xrange(1, len(MovieEndPoints)):
			DataPoint = random.randint(MovieEndPoints[Movie-1]+1, MovieEndPoints[Movie])
			RandomUser = SortedMovie[DataPoint,0]
			CurrMov = SortedMovie[DataPoint,1]
			ThetaJ = Theta[:,int(RandomUser-1)]
			Xi = X[:,int(CurrMov-1)]
			Yj = SortedMovie[DataPoint,2]
			AbsErr = numpy.dot(ThetaJ.T, Xi) - Yj
			#Grad = AbsErr*ThetaJ + Lambda*Xi
			Grad = AbsErr*ThetaJ + Lambda*numpy.sign(Xi)
			Alpha1 = Alpha*numpy.linalg.norm(Grad)
			Tmp = numpy.array([Xi - Alpha1*Grad])
			XNew[:,int(CurrMov-1)] = Tmp.T[:,0]

		#update Theta's
		for User in xrange(1, len(UserEndPoints)):
			DataPoint = random.randint(UserEndPoints[User-1]+1, UserEndPoints[User])
			RandomMovie = SortedUser[DataPoint,1]
			CurrUser = SortedUser[DataPoint,0]
			ThetaJ = Theta[:,int(CurrUser-1)]
			Xi = X[:,int(RandomMovie-1)]
			Yi = SortedUser[DataPoint,2]
			AbsErr = numpy.dot(ThetaJ.T, Xi) - Yi
			#Grad = AbsErr*Xi + Lambda*ThetaJ
			Grad = AbsErr*Xi + Lambda*numpy.sign(ThetaJ)
			Alpha1 = Alpha*numpy.linalg.norm(Grad)
			Tmp = numpy.array([ThetaJ - Alpha1*Grad])
			ThetaNew[:,int(CurrUser-1)] = Tmp.T[:,0]
		
		X = XNew
		Theta = ThetaNew
		if Graph:
			if i % SampleNum == 0:
				print i
				Ratings = numpy.dot(Theta.T,X)
				TestErr.append(square_error_collab(Ratings, TestMat))
				TrainErr.append(square_error_collab(Ratings, TrainMat))

	return (X, Theta, IterList, TestErr, TrainErr)

def stoch_grad_descent1(Theta, X, Iterations, Alpha, Lambda, SortedUser, SortedMovie, UserEndPoints, MovieEndPoints, TrainMat, TestMat, Graph, Model):
	XNew = X
	ThetaNew = Theta
	SampleNum = Iterations/100
	IterList = [i for i in xrange(0, Iterations+1, SampleNum)]
	TestErr = []
	TrainErr = []
	MovieIndex = 1
	UserIndex = 1
	for i in xrange(0, Iterations+1): 
		# update X's
		#Movie = random.randint(1, len(MovieEndPoints)-1)
		Movie = MovieIndex
		MovieIndex += 1
		if MovieIndex == len(MovieEndPoints): 
			MovieIndex = 1
		DataPoint = random.randint(MovieEndPoints[Movie-1]+1, MovieEndPoints[Movie])
		RandomUser = SortedMovie[DataPoint,0]
		CurrMov = SortedMovie[DataPoint,1]
		
		ThetaJ = Theta[:,int(RandomUser-1)]
		Xi = X[:,int(CurrMov-1)]
		Yj = SortedMovie[DataPoint,2]
		AbsErr = numpy.dot(ThetaJ.T, Xi) - Yj
		#L2
		if Model == 0:
			Grad = AbsErr*ThetaJ + Lambda*Xi
		#L1
		if Model == 1:
			Grad = AbsErr*ThetaJ + Lambda*numpy.sign(Xi)
		#ElasticNet
		#Grad = AbsErr*ThetaJ + Lambda*numpy.sign(Xi) + Lambda*Xi
		#Alpha1 = Alpha * numpy.linalg.norm(Grad)
		Alpha1 = Alpha 
		Tmp = numpy.array([Xi - Alpha1*Grad])
		XNew[:,int(CurrMov-1)] = Tmp.T[:,0]

		#update Theta's
		User = random.randint(1, len(UserEndPoints)-1)
		User = UserIndex
		UserIndex += 1 
		if UserIndex == len(UserEndPoints):
			UserIndex = 1
		DataPoint = random.randint(UserEndPoints[User-1]+1, UserEndPoints[User])
		RandomMovie = SortedUser[DataPoint,1]
		CurrUser = SortedUser[DataPoint,0]
		
		ThetaJ = Theta[:,int(CurrUser-1)]
		Xi = X[:,int(RandomMovie-1)]
		Yi = SortedUser[DataPoint,2]
		AbsErr = numpy.dot(ThetaJ.T, Xi) - Yi
		#L2
		if Model == 0:
			Grad = AbsErr*Xi + Lambda*ThetaJ
		#L1
		if Model == 1:
			Grad = AbsErr*Xi + Lambda*numpy.sign(ThetaJ)
		#ElasticNet
		#Grad = AbsErr*ThetaJ + Lambda*numpy.sign(Xi) + Lambda*Xi
		#Alpha1 = Alpha * numpy.linalg.norm(Grad)
		Alpha1 = Alpha 
		Tmp = numpy.array([ThetaJ - Alpha1*Grad])
		ThetaNew[:,int(CurrUser-1)] = Tmp.T[:,0]
		
		X = XNew
		Theta = ThetaNew
		if Graph:
			if i % SampleNum == 0:
				print i
				Ratings = numpy.dot(Theta.T,X) 
				TestErr.append(square_error_collab(Ratings, TestMat))
				TrainErr.append(square_error_collab(Ratings, TrainMat))

	return (X, Theta, IterList, TestErr, TrainErr)
	
def collaborative_filter(TrainMat, K, Lambda, Alpha, Iter, NumMovies, TestMat, Graph, Model):
	SortedUsers = data_ops.sort_col(TrainMat, 0)
	UserEndPoints = data_ops.extract_endpoints(SortedUsers, 0)
	NumUsers = len(UserEndPoints)
	UserEndPoints = [-1] + UserEndPoints

	SortedMovies = data_ops.sort_col(TrainMat, 1)
	MovieEndPoints = data_ops.extract_endpoints(SortedMovies, 1)
	MovieEndPoints = [-1] + MovieEndPoints

	X = numpy.random.rand(K, NumMovies)
	Theta = numpy.random.rand(K, 671)
	
	return stoch_grad_descent1(Theta, X, Iter, Alpha, Lambda, SortedUsers, SortedMovies, UserEndPoints, MovieEndPoints, TrainMat, TestMat, Graph, Model)

def n_fold_nested_cross_val(TrainMat, LambdaList, Alpha, KList, Iter, NumMovies, Fold, TestMat):
	BestValError = 1000000
	BestModel = () 

	for K in KList:
		for Lambda in LambdaList:
			TotalValError = 0
			for Index in xrange(0,TrainMat.shape[0]-Fold,Fold):
				TrainSet = TrainMat
				for Row in xrange(Index, Index+Fold):
					TrainSet = numpy.delete(TrainSet, (Index), axis=0)
				TestSet = TrainMat[Index:Index+Fold,:]
				(X, Theta,I,Te,Tr) = collaborative_filter(TrainSet, K, Lambda, Alpha, Iter, NumMovies, TestMat,False,0)
				Err = square_error_collab(numpy.dot(Theta.T, X), TestSet)
				print Err
				TotalValError += Err
			
			ValError = TotalValError / int(TrainMat.shape[0]/Fold)
			if ValError < BestValError : 
				BestValError = ValError
				BestModel = (Lambda,K)
			print "K: ", K, "Lambda: ", Lambda, "Err: ", BestValError

	return BestModel


def n_fold_cross_val(TrainMat, LambdaList, Alpha, KList, Iter, NumMovies, Fold, TestMat):
	BestValError = 1000000
	BestK = 1
	BestLambda = 1

	K = 8
	for Lambda in LambdaList:
		print "CurrLambda: ", Lambda
		TotalValError = 0
		for Index in xrange(0,TrainMat.shape[0]-Fold,Fold):
			TrainSet = TrainMat
			for Row in xrange(Index, Index+Fold):
				TrainSet = numpy.delete(TrainSet, (Index), axis=0)
			TestSet = TrainMat[Index:Index+Fold,:]
			(X, Theta,I,Te,Tr) = collaborative_filter(TrainSet, K, Lambda, Alpha, Iter, NumMovies, TestMat,False,0)
			Err = square_error_collab(numpy.dot(Theta.T, X), TestSet)
			print Err
			TotalValError += Err
		
		ValError = TotalValError / int(TrainMat.shape[0]/Fold)
		print "Avg Error: ", ValError
		if ValError < BestValError : 
			BestValError = ValError
			BestLambda = Lambda
	print "BestLambda: ", BestLambda
	
	Lambda = BestLambda
	BestValError = 1000000
	for K in KList:
		print "CurrK: ", K 
		TotalValError = 0
		for Index in xrange(0,TrainMat.shape[0]-Fold,Fold):
			TrainSet = TrainMat
			for Row in xrange(Index, Index+Fold):
				TrainSet = numpy.delete(TrainSet, (Index), axis=0)
			TestSet = TrainMat[Index:Index+Fold,:]
			(X, Theta,I,Te,Tr) = collaborative_filter(TrainSet, K, Lambda, Alpha, Iter, NumMovies, TestMat,False,0)
			Err = square_error_collab(numpy.dot(Theta.T, X), TestSet)
			print Err
			TotalValError += Err
		
		ValError = TotalValError / int(TrainMat.shape[0]/Fold)
		print "Avg Error: ", ValError
		if ValError < BestValError : 
			BestValError = ValError
			BestK = K
	print "BestK: ", BestK	
	
	#K = BestK
	#BestValError = 1000000
	#Lambda = 0.0001
	#for i in xrange(0,2):	
	#	print  "CurrModel: ", i
	#	TotalValError = 0
	#	for Index in xrange(0,TrainMat.shape[0]-Fold,Fold):
	#		TrainSet = TrainMat
	#		for Row in xrange(Index, Index+Fold):
	#			TrainSet = numpy.delete(TrainSet, (Index), axis=0)
	#		TestSet = TrainMat[Index:Index+Fold,:]
	#		(X, Theta,I,Te,Tr) = collaborative_filter(TrainSet, K, Lambda, Alpha, Iter, NumMovies, TestMat,False,i)
	#		Err = square_error_collab(numpy.dot(Theta.T, X), TestSet)
	#		print Err
	#		TotalValError += Err
	#	
	#	ValError = TotalValError / int(TrainMat.shape[0]/Fold)
	#	if ValError < BestValError : 
	#		BestValError = ValError
	#		BestModel = i
	#print "BestModel: ", BestModel


	return (BestLambda, BestK) 

