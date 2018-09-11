#include "em_util.h"

#include <fstream>
#include <vector>
#include <stdint.h>
#include <stdio.h>
#include <sys/time.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>


using Eigen::SimplicialLLT;
using Eigen::MatrixXd;
using namespace std;

// Returns difference in seconds
double toddiff(struct timeval *start, struct timeval *end) {
	long long tstart = start->tv_sec * 1000000 + start->tv_usec;
	long long tend = end->tv_sec * 1000000 + end->tv_usec;
	return ((double)(tend - tstart))/1000000.0;
}

double SoftThreshold(double a, double kappa) {
	return fmax(0, a-kappa) - fmax(0, -a-kappa);
}

double L1SubGrad(double x, double g, double lambda) {
	if (x > 0) {
		return g + lambda;
	} else if (x < 0) {
		return g - lambda;
	} else {
		return fmax(fabs(g) - lambda, 0);
	}
}

double L1Norm(const SpMatrixC& A) {
	double result = 0;
	for (long k = 0; k < A.outerSize(); ++k) {
		for (InIter it(A,k); it; ++it) {
			result += fabs(it.value());
		}
	}
	return result;
}

// Works with upper diag
double L1NormUpperDiag(const SpMatrixC& A) {
	double result = 0;
	for (long k = 0; k < A.outerSize(); ++k) {
		for (InIter it(A,k); it; ++it) {
			if (it.row() < it.col()) {
				result += 2*fabs(it.value());
			} else if (it.row() == it.col()) {
				result += fabs(it.value());
			} else {
				continue;
			}
		}
	}
	return result;
}

// Works with either upper diag or symmetric A
double L1NormOffDiag(const SpMatrixC& A) {
	double result = 0;
	for (long k = 0; k < A.outerSize(); ++k) {
		for (InIter it(A,k); it; ++it) {
			if (it.row() < it.col()) {
				result += 2*fabs(it.value());
			} else {
				continue;
			}
		}
	}
	return result;
}

// Interacts with Lambda as symmetric, not upper diag
double logdet(const SpMatrixC& Lambda) {
	SimplicialLLT<SpMatrixC> cholesky(Lambda);
	const SpMatrixC matrixL = cholesky.matrixL();
	return 2*matrixL.diagonal().array().log().sum();
}

double logdet(const SpMatrixC& matrixL, bool dummy) {
	return 2*matrixL.diagonal().array().log().sum();
}

// Computes trace(SA) = trace(X'*Y*A), where A is sparse matrix
double traceProduct(
		const SpMatrixC& A, const MatrixXd& X, const MatrixXd& Y) {
	long p = A.rows();
	long q = A.cols();
	long n = X.rows();
	// Complexity: nnz(A)*n
	
	double result = 0;
	for (long k = 0; k < A.outerSize(); ++k) {
		for (InIter it(A,k); it; ++it) {
			long i = it.row();
			long j = it.col();
			result += it.value() * X.col(i).dot(Y.col(j));
		}
	}
	return result;
}

// Computes trace(A' * B) = sum_ij(A_ij, B_ij)
double traceProduct(const MatrixXd& A, const MatrixXd& B) {
	return (A.cwiseProduct(B)).sum();
	/*
	long m = A.rows();
	long n = A.cols();
	// Complexity: mn
	
	double result = 0;
	for (long j = 0; j < n; ++j) { // assuming ColMajor order
		for (long i = 0; i < m; ++i) {
			result += A(i,j) * B(i,j);
		}
	}
	return result;
	*/
}

// Computes trace(A' * B)
double traceProductSp(const SpMatrixC& A, const MatrixXd& B) {
	long m = A.rows();
	long n = A.cols();
	
	double result = 0;
	for (long k = 0; k < n; ++k) {
		for (InIter it(A,k); it; ++it) {
			long i = it.row();
			long j = it.col();
			result += it.value() * B(i, j);
		}
	}
	return result;
}

bool readData(MatrixXd& Data, string Data_filename) {
	long n = Data.rows();
	long d = Data.cols();
	double val;
	ifstream ifD(Data_filename.c_str(), ifstream::in);
	for (long i = 0; i < n; i++) {
		for (long j = 0; j < d; j++) {
			if (!ifD.good()) {
				fprintf(stderr, "error reading %s\n", Data_filename.c_str());
				return false;
			}
			ifD >> val;
			Data(i,j) = val;
		}
	}
	ifD.close();
    return true;
}

void readLambda(SpMatrixC& Lambda, string Lambda_filename) {
	long q = Lambda.rows();
	ifstream ifL(Lambda_filename.c_str(), ifstream::in);
	long L_p, L_q, L_nnz;
	ifL >> L_p >> L_q >> L_nnz;
	if (L_p != q || L_q != q) {
		fprintf(stderr, "error reading Lambda_file\n");
		exit(1);
	}
	vector<TripE> triplets;
	long i, j;
	double val;
	for (long n = 0; n < L_nnz; n++) {
		ifL >> i >> j >> val;
		if (!ifL.good()) {
			fprintf(stderr, "error reading Lambda_file\n");
			exit(1);
		}
		if (i >= j) {
			triplets.push_back(TripE(i-1, j-1, val));
		}
	}
	Lambda.setFromTriplets(triplets.begin(), triplets.end());
	ifL.close();
} 

void readTheta(SpMatrixC& Theta, string Theta_filename) {
	long p = Theta.rows();
	long q = Theta.cols();
	ifstream ifT(Theta_filename.c_str(), ifstream::in);
	long T_p, T_q, T_nnz;
	ifT >> T_p >> T_q >> T_nnz;
	if (T_p != p || T_q != q) {
		fprintf(stderr, "error reading Theta_file\n");
		exit(1);
	}
	vector<TripE> triplets;
	long i, j;
	double val;
	for (long n = 0; n < T_nnz; n++) {
		ifT >> i >> j >> val;
		if (!ifT.good()) {
			fprintf(stderr, "error reading Theta_file\n");
			exit(1);
		}
		triplets.push_back(TripE(i-1, j-1, val));
	}
	Theta.setFromTriplets(triplets.begin(), triplets.end());
	ifT.close();
} 

void writeLambda(const SpMatrixC& Lambda, string Lambda_filename) {
	SpMatrixR Lambda_row = Lambda;
    long q = Lambda.rows();
	ofstream fL(Lambda_filename.c_str(), ofstream::out);
	fL.precision(12);
	long nnz_Lambda = 0;
	for (long k = 0; k < Lambda_row.outerSize(); k++) {
		for (InIterR it(Lambda_row, k); it; ++it) {
			if (it.value() != 0) {
				nnz_Lambda++;
			}
		}
	}
	fL << q << " " << q << " " << nnz_Lambda << endl;
	for (long k = 0; k < Lambda_row.outerSize(); k++) {
		for (InIterR it(Lambda_row, k); it; ++it) {
			if (it.value() != 0) {
				fL << it.row() + 1 << " " << it.col() + 1 << " " 
					<<  it.value() << endl;
			}
		}
	}
	fL.close();
}

void writeTheta(const SpMatrixC& Theta, string Theta_filename) {
    long p = Theta.rows();
    long q = Theta.cols();
	SpMatrixR Theta_row = Theta;
	ofstream fT(Theta_filename.c_str(), ofstream::out);
	fT.precision(12);
	long nnz_Theta = 0;
	for (long k = 0; k < Theta_row.outerSize(); k++) {
		for (InIterR it(Theta_row, k); it; ++it) {
			if (it.value() != 0) {
				nnz_Theta++;
			}
		}
	}
	
	fT << p << " " << q << " " << nnz_Theta << endl;
	for (long k = 0; k < Theta_row.outerSize(); k++) {
		for (InIterR it(Theta_row, k); it; ++it) {
			if (it.value() != 0) {
				fT << it.row() + 1 << " " << it.col() + 1 << " " 
					<< it.value() << endl;
			}
		}
	}
	fT.close();
}

void print(const SpMatrixC& A) {
	printf("p: %ld, nnz: %ld\n", A.outerSize(), A.nonZeros());
	for (long k = 0; k < A.outerSize(); ++k) {
		for (InIter it(A,k); it; ++it) {
			printf("%ld %ld %.10lf\n", it.row()+1, it.col()+1, it.value());
		}
	}
}

void print(const MatrixXd& A) {
	for (long i = 0; i < A.rows(); i++) {
		for (long j = 0; j < A.cols(); j++) {
			printf("%.2lf ", A(i,j));
		}
		printf("\n");
	}
}
