#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <math.h>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "fast_scggm.h"

using Eigen::VectorXi;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

void exit_with_help() {
	printf(
		"Usage: ./fast_scggm [options] "
		"Y_output_samples Y_output_dims X_input_samples X_input_dims Y_file X_file Lambda_file Theta_file stats_file\n"
		"options:\n"
		"    -y lambda_y(0.5): set the regularization parameter lambda_y\n"
		"    -x lambda_x(0.5): set the regularization parameter lambda_x\n"
		"    -v verbose(1): show information or not (0 or 1)\n"
		"    -i max_iters(10): max number of outer iterations\n"
		"    -s sigma(1e-4): backtracking termination criterion\n"
		"    -q tol(1e-2): tolerance for terminating outer loop\n"
		"    -L Lambda0_filename: filename with initial Lambda0\n"
		"    -T Theta0_filename: filename with initial Theta0\n"
		"    -r refit(false): update (Lambda0,Theta0) without adding edges\n"
	);
	exit(1);
}

int main(int argc, char **argv) {
	double lambda_y = 0.5;
	double lambda_x = 0.5;
	int num_reqd_args = 9;

	CGGMOptions options;
	if (argc < 1 + num_reqd_args) {
		fprintf(stderr,"not enough arguments\n");
		exit_with_help();
	}
	vector<string> cmdargs(argv + 1, argv + argc);
	int num_args = cmdargs.size();
	int num_opts_and_vals = num_args - num_reqd_args;
	int num_opts = (int) num_opts_and_vals / 2;
	if (num_opts_and_vals % 2 != 0) {
		fprintf(stderr,"option is missing a value\n");
		exit_with_help();
	}

	string Lambda0_filename = "";
	string Theta0_filename = "";

	for (int i = 0; i < num_opts; i++) {
		if (cmdargs[2*i][0] != '-') {
			fprintf(stderr,"incorrect option format\n");
			exit_with_help();
		}
		switch (cmdargs[2*i][1]) {
			case 'y':
				lambda_y = atof(cmdargs[2*i+1].c_str());
				break;
			case 'x':
				lambda_x = atof(cmdargs[2*i+1].c_str());
				break;
			case 'v':
				options.quiet = (atoi(cmdargs[2*i+1].c_str()) == 0);
				break;
			case 'i':
				options.max_outer_iters = atoi(cmdargs[2*i+1].c_str());
				break;
			case 's':
				options.sigma = atof(cmdargs[2*i+1].c_str());
				break;
			case 'q':
				options.tol = atof(cmdargs[2*i+1].c_str());
				break;
			case 'L':
				Lambda0_filename = cmdargs[2*i+1];
				break;
			case 'T':
				Theta0_filename = cmdargs[2*i+1];
				break;
			case 'r':
				options.refit = atoi(cmdargs[2*i+1].c_str()) != 0;
				break;
			default:
				fprintf(stderr,"unknown option: -%c\n", cmdargs[2*i][1]);
				exit_with_help();
				break;
		}
	}

	long n_y = atol(cmdargs[num_args-9].c_str());
	long q = atol(cmdargs[num_args-8].c_str());
	long n_x = atol(cmdargs[num_args-7].c_str());
	long p = atol(cmdargs[num_args-6].c_str());
	string Y_filename = cmdargs[num_args-5];
	string X_filename = cmdargs[num_args-4];
	string Lambda_filename = cmdargs[num_args-3];
	string Theta_filename = cmdargs[num_args-2];
	string stats_filename = cmdargs[num_args-1];
	if (!options.quiet) {
		fprintf(stdout, 
			"n_x=%li p=%li n_y=%li q=%li Yf=%s Xf=%s \nLf=%s Tf=%s sf=%s \n",
			n_x, p, n_y, q, Y_filename.c_str(), X_filename.c_str(), 
			Lambda_filename.c_str(), Theta_filename.c_str(), 
			stats_filename.c_str());
	}

	// Read input data from file
	MatrixXd Y(n_y, q);
	MatrixXd X(n_x, p);
	double val;
	ifstream ifY(Y_filename.c_str(), ifstream::in);
	for (long i = 0; i < n_y; i++) {
		for (long j = 0; j < q; j++) {
			if (!ifY.good()) {
				fprintf(stderr, "error reading Y_file\n");
				exit_with_help();
			}
			ifY >> val;
			Y(i,j) = val;
		}
	}
	ifY.close();
	ifstream ifX(X_filename.c_str(), ifstream::in);
	for (long i = 0; i < n_x; i++) {
		for (long j = 0; j < p; j++) {
			if (!ifX.good()) {
				fprintf(stderr, "error reading X_file\n");
				exit_with_help();
			}
			ifX >> val;
			X(i,j) = val;
		}
	}
	ifX.close();

	// Center and scale by 1/sqrt(n)
	//double scaling = 1.0/sqrt(n);
	VectorXd Y_mean = Y.colwise().mean();
	VectorXd X_mean = X.colwise().mean();
	Y.rowwise() -= Y_mean.transpose();
	X.rowwise() -= X_mean.transpose();
	//Y *= scaling;
	//X *= scaling;

	// Initialize sparse parameter matrices
	SpMatrixC Lambda(q, q);
	Lambda.reserve(VectorXi::Constant(q,1));
	for (long i = 0; i < q; ++i) {
		double variance = Y.col(i).dot(Y.col(i));
		if (variance <= 1.0e-8) {
			fprintf(stderr, "Y column %li has variance <= 1e-8 \n", i+1);
			exit(1);
		}
		Lambda.insert(i,i) = 1.0/(0.01 + (1.0/n_y)*variance);
	}
	SpMatrixC Theta(p, q);
	for (long i = 0; i < p; i++) {
		double variance = X.col(i).dot(X.col(i));
		if (variance <= 1.0e-8) {
			fprintf(stderr, "X column %li has variance <= 1e-8 \n", i+1);
			exit(1);
		}
	}

	// Initialize Lambda0 if specified by user
	if (!Lambda0_filename.empty()) {
		ifstream ifL(Lambda0_filename.c_str(), ifstream::in);
		long L0_p, L0_q, L0_nnz;
		ifL >> L0_p >> L0_q >> L0_nnz;
		if (L0_p != q || L0_q != q) {
			fprintf(stderr, "error reading Lambda0_file\n");
			exit(1);
		}
		vector<Triplet> triplets;
		long i, j;
		double val;
		for (long n = 0; n < L0_nnz; n++) {
			ifL >> i >> j >> val;
			if (!ifL.good()) {
				fprintf(stderr, "error reading Lambda0_file\n");
				exit(1);
			}
			if (i <= j) {
				triplets.push_back(Triplet(i-1, j-1, val));
			}
		}
		Lambda.setFromTriplets(triplets.begin(), triplets.end());
		ifL.close();
	}

	// Initialize Theta0 if specified by user
	if (!Theta0_filename.empty()) {
		ifstream ifT(Theta0_filename.c_str(), ifstream::in);
		long T0_p, T0_q, T0_nnz;
		ifT >> T0_p >> T0_q >> T0_nnz;
		if (T0_p != p || T0_q != q) {
			fprintf(stderr, "error reading Theta0_file\n");
			exit(1);
		}
		vector<Triplet> triplets;
		long i, j;
		double val;
		for (long n = 0; n < T0_nnz; n++) {
			ifT >> i >> j >> val;
			if (!ifT.good()) {
				fprintf(stderr, "error reading Theta0_file\n");
				exit(1);
			}
			triplets.push_back(Triplet(i-1, j-1, val));
		}
		Theta.setFromTriplets(triplets.begin(), triplets.end());
		ifT.close();
	}

	// Run optimization
	fflush(stdout);
	CGGMStats stats;
	fast_scggm(Y, X, lambda_y, lambda_x, options, Lambda, Theta, &stats);

	// Output sparse Lambda
	SpMatrixR Lambda_row = Lambda;
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

	// Output sparse Theta
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
	
	fT << p << " " << q << " " << Theta_row.nonZeros() << endl;
	for (long k = 0; k < Theta_row.outerSize(); k++) {
		for (InIterR it(Theta_row, k); it; ++it) {
			if (it.value() != 0) {
				fT << it.row() + 1 << " " << it.col() + 1 << " " 
					<< it.value() << endl;
			}
		}
	}
	fT.close();

	// Output stats
	ofstream fS(stats_filename.c_str(), ofstream::out);
	fS.precision(15);
	
	fS << "objval ";
	for (int i = 0; i < stats.objval.size(); i++) {
		fS << stats.objval[i] << " ";
	}
	fS << endl;
	fS << "time ";
	for (int i = 0; i < stats.time.size(); i++) {
		fS << stats.time[i] << " ";
	}
	fS << endl;
	fS << "active_set_size ";
	for (int i = 0; i < stats.active_set_size.size(); i++) {
		fS << stats.active_set_size[i] << " ";
	}
	fS << endl;
	fS << "active_theta ";
	for (int i = 0; i < stats.active_theta.size(); i++) {
		fS << stats.active_theta[i] << " ";
	}
	fS << endl;
	fS << "active_lambda ";
	for (int i = 0; i < stats.active_lambda.size(); i++) {
		fS << stats.active_lambda[i] << " ";
	}
	fS << endl;
	fS << "subgrad ";
	for (int i = 0; i < stats.subgrad.size(); i++) {
		fS << stats.subgrad[i] << " ";
	}
	fS << endl;
	fS << "l1norm ";
	for (int i = 0; i < stats.l1norm.size(); i++) {
		fS << stats.l1norm[i] << " ";
	}
	fS << endl;
	fS << "time_lambda_active ";
	for (int i = 0; i < stats.time_lambda_active.size(); i++) {
		fS << stats.time_lambda_active[i] << " ";
	}
	fS << endl;
	fS << "time_theta_active ";
	for (int i = 0; i < stats.time_theta_active.size(); i++) {
		fS << stats.time_theta_active[i] << " ";
	}
	fS << endl;
	fS << "time_lambda_cd ";
	for (int i = 0; i < stats.time_lambda_cd.size(); i++) {
		fS << stats.time_lambda_cd[i] << " ";
	}
	fS << endl;
	fS << "time_lambda_cd_cd ";
	for (int i = 0; i < stats.time_lambda_cd_cd.size(); i++) {
		fS << stats.time_lambda_cd_cd[i] << " ";
	}
	fS << endl;
	fS << "time_lambda_cd_ls ";
	for (int i = 0; i < stats.time_lambda_cd_ls.size(); i++) {
		fS << stats.time_lambda_cd_ls[i] << " ";
	}
	fS << endl;
	fS << "time_theta_cd ";
	for (int i = 0; i < stats.time_theta_cd.size(); i++) {
		fS << stats.time_theta_cd[i] << " ";
	}
	fS << endl;
	fS << "time_theta_cd_cd ";
	for (int i = 0; i < stats.time_theta_cd_cd.size(); i++) {
		fS << stats.time_theta_cd_cd[i] << " ";
	}
	fS << endl;
	fS << "time_theta_cd_qr ";
	for (int i = 0; i < stats.time_theta_cd_qr.size(); i++) {
		fS << stats.time_theta_cd_qr[i] << " ";
	}
	fS << endl;

	fS.close();
	return 0;
}

