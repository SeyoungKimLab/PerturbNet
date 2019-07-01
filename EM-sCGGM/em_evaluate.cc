#include "em_objective.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <math.h>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>

typedef Eigen::SparseMatrix<double,Eigen::ColMajor,long int> SpMatrixC;
typedef SpMatrixC::InnerIterator InIter;
typedef Eigen::Triplet<double> TripE;

using Eigen::VectorXi;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Upper;
using Eigen::SimplicialLLT;

using namespace std;

void exit_with_help() {
	printf(
		"Usage: ./em_evaluate [options] "
		"Z_dims Y_dims X_dims XZ_samples Y_samples Z_testfile Y_testfile X_testfile Lambda_zz_file Theta_yz_file Lambda_yy_file Theta_xy_file eval_file\n"
		"options:\n"
		"    -Z lambda_zz : set the regularization for Lambda_z (default 0.5)\n"
		"    -z lambda_yz : set the regularization for Theta_yz (default 0.5)\n"
		"    -Y lambda_yy : set the regularization for Lambda_y (default 0.5)\n"
		"    -y lambda_xy : set the regularization for Theta_xy (default 0.5)\n"
	);
	exit(1);
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

double L1Norm(const SpMatrixC& A) {
	double result = 0;
	for (long k = 0; k < A.outerSize(); ++k) {
		for (InIter it(A,k); it; ++it) {
			result += fabs(it.value());
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

void readSparseSymmetricEigen(SpMatrixC& T, string fname) {
	ifstream ifs(fname.c_str(), ifstream::in);
	long p, q, nnz;
	ifs >> p >> q >> nnz;
	if (p != T.rows() || q != T.cols()) {
		fprintf(stderr, "error %s dimensions\n", fname.c_str());
		exit(1);
	}
	long i, j;
	double val;
	vector<TripE> triplets;
	for (long nz = 0; nz < nnz; nz++) {
		ifs >> i >> j >> val;
		if (!ifs.good()) {
			fprintf(stderr, "error reading %s\n", fname.c_str());
			exit(1);
		}
		if (i <= j) {
			triplets.push_back(TripE(i-1, j-1, val));
		}
	}
	T.setFromTriplets(triplets.begin(), triplets.end());
	ifs.close();
}

void readSparseEigen(SpMatrixC& T, string fname) {
	ifstream ifs(fname.c_str(), ifstream::in);
	long p, q, nnz;
	ifs >> p >> q >> nnz;
	if (p != T.rows() || q != T.cols()) {
		fprintf(stderr, "error %s dimensions\n", fname.c_str());
		exit(1);
	}
	long i, j;
	double val;
	vector<TripE> triplets;
	for (long nz = 0; nz < nnz; nz++) {
		ifs >> i >> j >> val;
		if (!ifs.good()) {
			fprintf(stderr, "error reading %s\n", fname.c_str());
			exit(1);
		}
		triplets.push_back(TripE(i-1, j-1, val));
	}
	T.setFromTriplets(triplets.begin(), triplets.end());
	ifs.close();
}

int main(int argc, char **argv) {
	double lambda_zz = 0.1;
	double lambda_yz = 0.1;
	double lambda_yy = 0.1;
	double lambda_xy = 0.1;

	int num_reqd_args = 12;
	vector<string> cmdargs(argv + 1, argv + argc);
	int num_args = cmdargs.size();
	int num_opts_and_vals = num_args - num_reqd_args;
	int num_opts = (int) num_opts_and_vals / 2;
	if (num_args < num_reqd_args) {
		fprintf(stderr,"missing args \n");
		exit_with_help();
	}	
	if (num_opts_and_vals % 2 != 0) {
		fprintf(stderr,"option is missing a value\n");
		exit_with_help();
	}

	long r = atol(cmdargs[num_args-12].c_str());
	long q = atol(cmdargs[num_args-11].c_str());
	long p = atol(cmdargs[num_args-10].c_str());
	long n = atol(cmdargs[num_args-9].c_str());
	long n_o = atol(cmdargs[num_args-8].c_str());
	string Z_file = cmdargs[num_args-7];
	string Y_file = cmdargs[num_args-6];
	string X_file = cmdargs[num_args-5];
	string Lambda_zz_file = cmdargs[num_args-4];
	string Theta_yz_file = cmdargs[num_args-3];
	string Lambda_yy_file = cmdargs[num_args-2];
	string Theta_xy_file = cmdargs[num_args-1];

	if (argc < 1 + num_reqd_args) {
		fprintf(stderr,"not enough arguments\n");
		exit_with_help();
	}

	for (int i = 0; i < num_opts; i++) {
		if (cmdargs[2*i][0] != '-') {
			fprintf(stderr,"incorrect option format\n");
			exit_with_help();
		}
		switch (cmdargs[2*i][1]) {
			case 'Z':
				lambda_zz = atof(cmdargs[2*i+1].c_str());
				break;
			case 'z':
				lambda_yz = atof(cmdargs[2*i+1].c_str());
				break;
			case 'Y':
				lambda_yy = atof(cmdargs[2*i+1].c_str());
				break;
			case 'y':
				lambda_xy = atof(cmdargs[2*i+1].c_str());
				break;
			default:
				fprintf(stderr,"unknown option: -%c\n", cmdargs[2*i][1]);
				exit_with_help();
				break;
		}
	}
	fprintf(stdout, 
		"r=%li q=%li p=%li n=%li n_o=%li "
		"lambdaLzz:%.5f lambdaTyz:%.5f lambdaLyy:%.5f lambdaTxy:%.5f \n"
		"Zf=%s \nYf=%s \nXf=%s \n"
		"Lzzf=%s \nTyzf=%s \nLyyf=%s \nTxyf=%s \n",
		r, q, p, n, n_o, lambda_zz, lambda_yz, lambda_yy, lambda_xy,
		Z_file.c_str(), Y_file.c_str(), X_file.c_str(), 
		Lambda_zz_file.c_str(), Theta_yz_file.c_str(), 
		Lambda_yy_file.c_str(), Theta_xy_file.c_str());


	// Read input data from files
	MatrixXd Z(n, r);
	if (!readData(Z, Z_file)) {
		exit_with_help();
	}
	MatrixXd Y_o(n_o, q);
	if (!readData(Y_o, Y_file)) {
		exit_with_help();
	}
	MatrixXd X(n, p);
	if (!readData(X, X_file)) {
		exit_with_help();
	}
	// Center and scale by 1/sqrt(n)
	// To match with obj_min, do not center or scale
	VectorXd Z_mean = Z.colwise().mean();
	VectorXd Y_mean = Y_o.colwise().mean();
	VectorXd X_mean = X.colwise().mean();
	//Z.rowwise() -= Z_mean.transpose();
	//Y_o.rowwise() -= Y_mean.transpose();
	//X.rowwise() -= X_mean.transpose();
	//double scaling = 1.0/sqrt(n);
	//Z *= scaling;
	//Y *= scaling;
	//X *= scaling;

	// Read parameter matrices from files
	SpMatrixC Lambda_zz_sym(r, r);
	SpMatrixC Theta_yz(q, r);
	SpMatrixC Lambda_yy_sym(q, q);
	SpMatrixC Theta_xy(p, q);
	readSparseEigen(Lambda_zz_sym, Lambda_zz_file);
	readSparseEigen(Theta_yz, Theta_yz_file);
	readSparseEigen(Lambda_yy_sym, Lambda_yy_file);
	readSparseEigen(Theta_xy, Theta_xy_file);

	SpMatrixC Lambda_zz(r, r);
	SpMatrixC Lambda_yy(q, q);
	readSparseSymmetricEigen(Lambda_zz, Lambda_zz_file);
	readSparseSymmetricEigen(Lambda_yy, Lambda_yy_file);
	Lambda_zz_sym = Lambda_zz.selfadjointView<Upper>();
	Lambda_yy_sym = Lambda_yy.selfadjointView<Upper>();

	// Evaluate objective
	double obj = EMObjective(Z, Y_o, X, 
		lambda_zz, lambda_yz, lambda_yy, lambda_xy,
		Lambda_zz_sym, Theta_yz, Lambda_yy_sym, Theta_xy);
	printf("obj=%.7f \n", obj);
	fflush(stdout);
	return 0;
}

