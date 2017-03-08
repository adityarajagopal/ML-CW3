import csv
import numpy
import random
import scipy
from sklearn import decomposition

numpy.set_printoptions(threshold = numpy.nan)
def reader(File):
	Data = numpy.loadtxt(open(File,"rb"), delimiter=",", skiprows=1)
	return Data

def sort_col(Matrix, Col):
	return Matrix[Matrix[:,Col].argsort()]
	
def mean_by_col(Matrix, Col, DataCol, TrueTotal):
	PrevMovie = 1
	Count = 0
	CumRating = 0
	AvgRating = numpy.array([[0,0]]) 
	for Row in Matrix:
		if (Row[Col] == PrevMovie):
			CumRating += Row[DataCol]
			Count += 1
		else:
			Avg = CumRating / Count 
			AvgRating = numpy.append(AvgRating, [[PrevMovie,Avg]], axis=0)
			if Row[Col] != PrevMovie+1:
				AvgRating = append_zero_rows(PrevMovie+1, Row[Col], AvgRating)
			PrevMovie = Row[Col]
			CumRating = Row[DataCol]
			Count = 1
	
	Avg = CumRating / Count
	AvgRating = numpy.append(AvgRating, [[PrevMovie,Avg]], axis=0)
	
	if PrevMovie != TrueTotal:
		AvgRating = append_zero_rows(PrevMovie+1, TrueTotal+1, AvgRating)

	AvgRating = numpy.delete(AvgRating, (0), axis=0)
	
	return AvgRating		

def append_zero_rows(Start, End, Matrix):
	for i in xrange(int(Start),int(End)):
		Matrix = numpy.append(Matrix, [[i,0]], axis=0)
	return Matrix

def extract_endpoints(Matrix, Col):
	PrevElem = Matrix[0,Col]
	Count = 0
	EndPoints = []
	for Elem in Matrix[:,Col]:
		if Elem != PrevElem:
			PrevElem = Elem
			EndPoints.append(Count-1)
		Count += 1
	EndPoints.append(Count-1)
	return EndPoints			

def normalise(Matrix):
	ColMeans = numpy.mean(Matrix, axis=0)
	ColSd = numpy.std(Matrix, axis=0)
	return (Matrix - ColMeans)/ColSd

def pca(FeatMat, NumFeats):
	PCA = decomposition.PCA(n_components = NumFeats)
	Z = PCA.fit_transform(FeatMat)
	return Z

def legendre(FeatMat, Degree):	
	Z = numpy.zeros((FeatMat.shape[0], 1))
	for Deg in xrange(0, Degree+1):
		L = numpy.polyval(scipy.special.legendre(Deg), FeatMat)
		Z = numpy.append(Z, L, axis=1)
	Z = numpy.delete(Z, (0), axis=1)
	return Z
		
	








