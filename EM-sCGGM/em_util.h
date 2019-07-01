#ifndef _EM_UTIL_H_
#define _EM_UTIL_H_

#include <iostream>
#include <vector>
#include <stdint.h>
#include <stdio.h>
#include <sys/time.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

typedef Eigen::SparseMatrix<double,Eigen::ColMajor,long int> SpMatrixC;
typedef Eigen::SparseMatrix<double,Eigen::RowMajor,long int> SpMatrixR;
typedef SpMatrixC::InnerIterator InIter;
typedef SpMatrixR::InnerIterator InIterR;
typedef Eigen::Triplet<double> TripE;

using Eigen::MatrixXd;
using std::string;

// Returns difference in seconds
double toddiff(struct timeval *start, struct timeval *end); 

double SoftThreshold(double a, double kappa);
double L1SubGrad(double x, double g, double lambda);

double L1Norm(const SpMatrixC& A);

// Works with upper diag
double L1NormUpperDiag(const SpMatrixC& A);

// Works with either upper diag or symmetric A
double L1NormOffDiag(const SpMatrixC& A);

// Interacts with Lambda as symmetric, not upper diag
double logdet(const SpMatrixC& Lambda);

double logdet(const SpMatrixC& matrixL, bool dummy);

// Uses Conjugate Gradient (CG) method
int logdetDense(const Eigen::MatrixXd& A, double tol, double& result);

// Computes trace(SA) = trace(X'*Y*A), where A is sparse matrix
double traceProduct(
		const SpMatrixC& A, const MatrixXd& X, const MatrixXd& Y);

// Computes trace(A' * B) = sum_ij(A_ij, B_ij)
double traceProduct(const MatrixXd& A, const MatrixXd& B);

// Computes trace(A' * B)
double traceProductSp(const SpMatrixC& A, const MatrixXd& B);

bool readData(MatrixXd& Data, string Data_filename);

void readLambda(SpMatrixC& Lambda, string Lambda_filename);

void readTheta(SpMatrixC& Theta, string Theta_filename);

void writeLambda(const SpMatrixC& Lambda, string Lambda_filename);

void writeTheta(const SpMatrixC& Theta, string Theta_filename);

void print(const SpMatrixC& A);
void print(const MatrixXd& A);

#endif
