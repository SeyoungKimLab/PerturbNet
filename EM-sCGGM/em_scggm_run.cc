#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <math.h>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "em_util.h"
#include "em_scggm.h"

using Eigen::VectorXi;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Upper;

using namespace std;

void exit_with_help() {
	printf(
		"Usage: ./em_scggm [options] "
		"Z_dims Y_dims X_dims XZ_samples Y_samples Z_file Y_file X_file Lambda_z_file Theta_yz_file Lambda_y_file Theta_xy_file stats_file\n"
		"options:\n"
		"    -Z lambda_zz : set the regularization for Lambda_z (default 0.5)\n"
		"    -z lambda_yz : set the regularization for Theta_yz (default 0.5)\n"
		"    -Y lambda_yy : set the regularization for Lambda_y (default 0.5)\n"
		"    -y lambda_xy : set the regularization for Theta_xy (default 0.5)\n"
		"    -v verbose(1): show information or not (0 / 1 / 2)\n"
		"    -I max_em_iters(20) : max number of EM iterations\n"
		"    -i max_M_iters(1) : max number of iterations within M-step\n"
		"    -E sparse_E_step(1) : avoid dense inverse in E step (0 or 1)\n"
		"    -T sparse_Theta_cd(1) : avoid dense products in ThetaCD (0 or 1)\n"
		"    -s sigma: backtracking termination criterion\n"
		"    -Q em_tol: tolerance for terminating EM loop\n"
		"    -q tol: tolerance for terminating M-step\n"
		"    -l num_blocks_Lambda(-1): number of blocks for Lambda CD\n"
		"    -t num_blocks_Theta(-1): number of blocks for Theta CD\n"
		"    -m memory_usage(32000): memory capacity in MB\n" 
		"    -n threads(16) : set the max number of threads\n"    
	);
	exit(1);
}

int main(int argc, char **argv) {
	double lambda_zz = 0.5;
	double lambda_yz = 0.5;
	double lambda_yy = 0.5;
	double lambda_xy = 0.5;

	int num_reqd_args = 13;
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

	long r = atol(cmdargs[num_args-13].c_str());
	long q = atol(cmdargs[num_args-12].c_str());
	long p = atol(cmdargs[num_args-11].c_str());
	long n = atol(cmdargs[num_args-10].c_str());
	long n_o = atol(cmdargs[num_args-9].c_str());
	string Z_file = cmdargs[num_args-8];
	string Y_file = cmdargs[num_args-7];
	string X_file = cmdargs[num_args-6];
	string Lambda_zz_file = cmdargs[num_args-5];
	string Theta_yz_file = cmdargs[num_args-4];
	string Lambda_yy_file = cmdargs[num_args-3];
	string Theta_xy_file = cmdargs[num_args-2];
	string stats_file = cmdargs[num_args-1];

	fprintf(stdout, "r=%li q=%li p=%li n=%li n_o=%li \n"
		"Zf=%s Yf=%s Xf=%s Lzzf=%s Tyzf=%s Lyyf=%s Txyf=%s sf=%s \n",
		r, q, p, n, n_o, Z_file.c_str(), Y_file.c_str(), X_file.c_str(), 
		Lambda_zz_file.c_str(), Theta_yz_file.c_str(), 
		Lambda_yy_file.c_str(), Theta_xy_file.c_str(), stats_file.c_str());

	EMOptions options;
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
			case 'v':
				options.verbose = atoi(cmdargs[2*i+1].c_str());
				break;
			case 'I':
				options.max_em_iters = atoi(cmdargs[2*i+1].c_str());
				break;
			case 'i':
				options.max_M_iters = atoi(cmdargs[2*i+1].c_str());
				break;
			case 'E':
				options.sparse_E_step = (atoi(cmdargs[2*i+1].c_str()) == 1);
				break;
			case 'T':
				options.sparse_Theta_cd = (atoi(cmdargs[2*i+1].c_str()) == 1);
				break;
			case 's':
				options.sigma = atof(cmdargs[2*i+1].c_str());
				break;
			case 'Q':
				options.em_tol = atof(cmdargs[2*i+1].c_str());
				break;
			case 'q':
				options.tol = atof(cmdargs[2*i+1].c_str());
				break;
			case 'l':
				options.num_blocks_Lambda = atol(cmdargs[2*i+1].c_str());
				break;
			case 't':
				options.num_blocks_Theta = atol(cmdargs[2*i+1].c_str());
				break;
			case 'm':
				options.memory_usage = atol(cmdargs[2*i+1].c_str());
				break;
			case 'n':
				options.max_threads = atoi(cmdargs[2*i+1].c_str());
				break;
			default:
				fprintf(stderr,"unknown option: -%c\n", cmdargs[2*i][1]);
				exit_with_help();
				break;
		}
	}

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
	if (options.verbose >= 2) {
		printf("EM-sCGGM done reading input data \n");
		fflush(stdout);
	}
	// Center and scale by 1/sqrt(n)
	VectorXd Z_mean = Z.colwise().mean();
	VectorXd Y_mean = Y_o.colwise().mean();
	VectorXd X_mean = X.colwise().mean();
	Z.rowwise() -= Z_mean.transpose();
	Y_o.rowwise() -= Y_mean.transpose();
	X.rowwise() -= X_mean.transpose();
	//double scaling = 1.0/sqrt(n);
	//Z *= scaling;
	//Y *= scaling;
	//X *= scaling;

	// Initialize sparse parameter matrices
	SpMatrixC Lambda_zz(r, r);
	Lambda_zz.reserve(VectorXi::Constant(r,1));
	for (long i = 0; i < r; ++i) {
		double variance = Z.col(i).dot(Z.col(i));
		if (variance <= 1.0e-8) {
			fprintf(stderr, "Z column %li has variance <= 1e-8 \n", i+1);
			exit(1);
		}
		Lambda_zz.insert(i,i) = 1.0/(0.01 + (1.0/n)*variance);
	}
	SpMatrixC Theta_yz(q, r);
	SpMatrixC Lambda_yy(q, q);
	Lambda_yy.reserve(VectorXi::Constant(q,1));
	for (long i = 0; i < q; ++i) {
		double variance = Y_o.col(i).dot(Y_o.col(i));
		if (variance <= 1.0e-8) {
			fprintf(stderr, "Y_o column %li has variance <= 1e-8 \n", i+1);
			exit(1);
		}
		Lambda_yy.insert(i,i) = 1.0/(0.01 + (1.0/n_o)*variance);
	}
	SpMatrixC Theta_xy(p, q);
	for (long i = 0; i < p; i++) {
		double variance = X.col(i).dot(X.col(i));
		if (variance <= 1.0e-8) {
			fprintf(stderr, "X column %li has variance <= 1e-8 \n", i+1);
			exit(1);
		}
	}

	// Run optimization
	fflush(stdout);
	EMStats stats;
	em_scggm(Z, Y_o, X, lambda_zz, lambda_yz, lambda_yy, lambda_xy, 
		Lambda_zz, Theta_yz, Lambda_yy, Theta_xy, options, &stats);


	// Write sparse parameters
	SpMatrixC Lambda_zz_sym;
	Lambda_zz_sym = Lambda_zz.selfadjointView<Upper>();
	Lambda_zz = Lambda_zz_sym;
	SpMatrixC Lambda_yy_sym;
	Lambda_yy_sym = Lambda_yy.selfadjointView<Upper>();
	Lambda_yy = Lambda_yy_sym;
	writeLambda(Lambda_zz, Lambda_zz_file);
	writeTheta(Theta_yz, Theta_yz_file);
	writeLambda(Lambda_yy, Lambda_yy_file);
	writeTheta(Theta_xy, Theta_xy_file);

	// Output stats
	ofstream fS(stats_file.c_str(), ofstream::out);
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
	fS << "l1norm ";
	for (int i = 0; i < stats.l1norm.size(); i++) {
		fS << stats.l1norm[i] << " ";
	}
	fS << endl;
	fS << "iters_yz ";
	for (int i = 0; i < stats.iters_yz.size(); i++) {
		fS << stats.iters_yz[i] << " ";
	}
	fS << endl;
	fS << "iters_xy ";
	for (int i = 0; i < stats.iters_xy.size(); i++) {
		fS << stats.iters_xy[i] << " ";
	}
	fS << endl;

	fS << "active_lambda_zz ";
	for (int i = 0; i < stats.active_lambda_zz.size(); i++) {
		fS << stats.active_lambda_zz[i] << " ";
	}
	fS << endl;
	fS << "active_theta_yz ";
	for (int i = 0; i < stats.active_theta_yz.size(); i++) {
		fS << stats.active_theta_yz[i] << " ";
	}
	fS << endl;
	fS << "active_lambda_yy ";
	for (int i = 0; i < stats.active_lambda_yy.size(); i++) {
		fS << stats.active_lambda_yy[i] << " ";
	}
	fS << endl;
	fS << "active_theta_xy ";
	for (int i = 0; i < stats.active_theta_xy.size(); i++) {
		fS << stats.active_theta_xy[i] << " ";
	}
	fS << endl;

	fS << "time_e_step ";
	for (int i = 0; i < stats.time_e_step.size(); i++) {
		fS << stats.time_e_step[i] << " ";
	}
	fS << endl;
	fS << "time_lambda_zz_active ";
	for (int i = 0; i < stats.time_lambda_zz_active.size(); i++) {
		fS << stats.time_lambda_zz_active[i] << " ";
	}
	fS << endl;
	fS << "time_lambda_zz_cd ";
	for (int i = 0; i < stats.time_lambda_zz_cd.size(); i++) {
		fS << stats.time_lambda_zz_cd[i] << " ";
	}
	fS << endl;
	fS << "time_theta_yz_active ";
	for (int i = 0; i < stats.time_theta_yz_active.size(); i++) {
		fS << stats.time_theta_yz_active[i] << " ";
	}
	fS << endl;
	fS << "time_theta_yz_cd ";
	for (int i = 0; i < stats.time_theta_yz_cd.size(); i++) {
		fS << stats.time_theta_yz_cd[i] << " ";
	}
	fS << endl;
	
	fS << "time_lambda_yy_active ";
	for (int i = 0; i < stats.time_lambda_yy_active.size(); i++) {
		fS << stats.time_lambda_yy_active[i] << " ";
	}
	fS << endl;
	fS << "time_lambda_yy_cd ";
	for (int i = 0; i < stats.time_lambda_yy_cd.size(); i++) {
		fS << stats.time_lambda_yy_cd[i] << " ";
	}
	fS << endl;
	fS << "time_theta_xy_active ";
	for (int i = 0; i < stats.time_theta_xy_active.size(); i++) {
		fS << stats.time_theta_xy_active[i] << " ";
	}
	fS << endl;
	fS << "time_theta_xy_cd ";
	for (int i = 0; i < stats.time_theta_xy_cd.size(); i++) {
		fS << stats.time_theta_xy_cd[i] << " ";
	}
	fS << endl;

	fS.close();
	return 0;
}

