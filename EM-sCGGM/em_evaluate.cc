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
		"Usage: ./em_evaluate_run [options] "
		"r q p n n_o Z_testfile Y_testfile X_testfile Lambda_zz_file Theta_yz_file Lambda_yy_file Theta_xy_file eval_file\n"
		"options:\n"
		"    -Z lambda_zz : set the regularization parameter lambda_zz (default 0.5)\n"
		"    -z lambda_yz : set the regularization parameter lambda_zy (default 0.5)\n"
		"    -Y lambda_yy : set the regularization parameter lambda_yy (default 0.5)\n"
		"    -y lambda_xy : set the regularization parameter lambda_xy (default 0.5)\n"
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

double EMObjective(
		const Eigen::MatrixXd& Z,
		const Eigen::MatrixXd& Y_o,
		const Eigen::MatrixXd& X,
		double lambda_zz,
		double lambda_yz,
		double lambda_yy,
		double lambda_xy,
		SpMatrixC& Lambda_zz, // still works if Lambda_zz_sym
		SpMatrixC& Theta_yz,
		SpMatrixC& Lambda_yy, // stil works if Lambda_yy_sym
		SpMatrixC& Theta_xy) {
	long n = Z.rows();
	long n_o = Y_o.rows();
	long n_h = n - n_o;
	long p = X.cols();
	long q = Y_o.cols();
	long r = Z.cols();
	MatrixXd X_o = X.topRows(n_o);
	MatrixXd Z_o = Z.topRows(n_o);
	MatrixXd X_h = X.bottomRows(n_h);
	MatrixXd Z_h = Z.bottomRows(n_h);
	SpMatrixC Lambda_zz_sym;
	Lambda_zz_sym = Lambda_zz.selfadjointView<Upper>();
	SpMatrixC Lambda_yy_sym;
	Lambda_yy_sym = Lambda_yy.selfadjointView<Upper>();
	SimplicialLLT<SpMatrixC> chol_zz(Lambda_zz_sym);
	SimplicialLLT<SpMatrixC> chol_yy(Lambda_yy_sym);
	MatrixXd XoThT = (-1*X_o*Theta_xy).transpose(); // q x n_o
	MatrixXd YoThT = (-1*Y_o*Theta_yz).transpose(); // r x n_o
	MatrixXd Y_XB = Y_o - chol_yy.solve(XoThT).transpose(); // n_o x q
	MatrixXd Z_YB = Z_o - chol_zz.solve(YoThT).transpose();	// n_o x r
	double ll_xy = traceProduct(Lambda_yy_sym, Y_XB, Y_XB) - 
		n_o*logdet(chol_yy.matrixL(), true);
	double ll_yz = traceProduct(Lambda_zz_sym, Z_YB, Z_YB) -
		n_o*logdet(chol_zz.matrixL(), true);

	MatrixXd XhThT = (-1*X_h*Theta_xy).transpose();   // n_h x q
	MatrixXd eY_h = chol_yy.solve(XhThT).transpose(); // n_h x q
	MatrixXd YhThT = (-1*eY_h*Theta_yz).transpose();  // n_h x r
	MatrixXd eZX_h = chol_zz.solve(YhThT).transpose(); // n_h x r
	MatrixXd Z_Xmu_h = Z_h - eZX_h; // n_h x r
	MatrixXd Ir = MatrixXd::Identity(r, r);
	MatrixXd Sig_z = chol_zz.solve(Ir);
	MatrixXd Sig_zx = (Ir + Sig_z*(
		Theta_yz.transpose()*chol_yy.solve(Theta_yz)))*Sig_z;
	Eigen::PartialPivLU<MatrixXd> Sig_zx_decomp(Sig_zx);
	MatrixXd iSig_zx = Sig_zx_decomp.solve(Ir);
	MatrixXd innerZ_Xmu_h = Z_Xmu_h.transpose()*Z_Xmu_h;
	double ll_xz = traceProduct(iSig_zx, innerZ_Xmu_h) +
		n_h*log(Sig_zx_decomp.determinant());
	double pen = lambda_zz*L1NormOffDiag(Lambda_zz_sym) + 
		lambda_yz*L1Norm(Theta_yz) + lambda_yy*L1NormOffDiag(Lambda_yy_sym) +
		lambda_xy*L1Norm(Theta_xy);
	return (ll_xz + ll_xy + ll_yz)/n + pen;
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
	long i, j;
	double val;
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

