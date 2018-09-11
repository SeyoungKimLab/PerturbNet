#ifndef _EM_SCGGM_H_
#define _EM_SCGGM_H_

#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>

typedef Eigen::SparseMatrix<double,Eigen::ColMajor,long int> SpMatrixC;

using Eigen::MatrixXd;
using std::vector;

struct EMOptions {
		EMOptions() : 
	quiet(false),
	max_em_iters(20),
	max_outer_iters(1),
	max_Theta_iters(1),
	max_Lambda_iters(1),
    max_ls_iters(10),
    sparse_E_step(true),
    sparse_Theta_cd(true),
    alpha(1),
    beta(0.5),
    sigma(1.0e-4),
	em_tol(1.0e-3),
    tol(1.0e-4),
 	num_blocks_Theta(-1), // <0: min with memory, >0: user input
 	num_blocks_Lambda(-1), // <0: min with memory, >0: user input
	memory_usage(32000) {}
	
	// Whether to print info messages
	bool quiet;

	// Maximum iterations 
	int max_em_iters;
	int max_outer_iters;
	int max_Theta_iters;
	int max_Lambda_iters;
	int max_ls_iters;

	// Sparse computations parameters
	bool sparse_E_step;
	bool sparse_Theta_cd;

	// Line search parameters
	double alpha;
	double beta;
	double sigma;

	// Tolerance parameters 
	double em_tol;
	double tol;

	// Limited-memory parameters
	long num_blocks_Theta;
	long num_blocks_Lambda;
	long memory_usage;
};

struct EMStats {
	vector<double> objval;
	vector<double> time;
	vector<double> l1norm;
	vector<double> iters_yz;
	vector<double> iters_xy;

	vector<double> active_lambda_zz;
	vector<double> active_theta_yz;
	vector<double> active_lambda_yy;
	vector<double> active_theta_xy;

	vector<double> time_e_step;
	vector<double> time_lambda_zz_active;
	vector<double> time_lambda_zz_cd;
	vector<double> time_theta_yz_active;
	vector<double> time_theta_yz_cd;
	vector<double> time_lambda_yy_active;
	vector<double> time_lambda_yy_cd;
	vector<double> time_theta_xy_active;
	vector<double> time_theta_xy_cd;
};

struct CGGMStats {
	std::vector<double> objval;
	std::vector<double> time;
	std::vector<double> active_set_size;
	std::vector<double> active_theta;
	std::vector<double> active_lambda;
    std::vector<double> subgrad;
	std::vector<double> l1norm;

	std::vector<double> time_lambda_active;
	std::vector<double> time_theta_active;
	std::vector<double> time_theta_cd;
	std::vector<double> time_theta_cd_cd;
	std::vector<double> time_theta_cd_qr;
	std::vector<double> time_lambda_cd;
	std::vector<double> time_lambda_cd_cd;
	std::vector<double> time_lambda_cd_ls;
};

void em_scggm(
    const Eigen::MatrixXd& Z,
	const Eigen::MatrixXd& Y_o,
	const Eigen::MatrixXd& X,
	double lambda_zz,
	double lambda_yz,
	double lambda_yy,
	double lambda_xy,
	SpMatrixC& Lambda_zz,
	SpMatrixC& Theta_yz,
	SpMatrixC& Lambda_yy,
	SpMatrixC& Theta_xy,
	EMOptions& options,
	EMStats* stats);

#endif
