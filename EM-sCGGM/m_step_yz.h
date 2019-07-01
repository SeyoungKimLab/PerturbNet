#ifndef _M_STEP_YZ_H_
#define _M_STEP_YZ_H_

#include "em_scggm.h"

#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>

typedef Eigen::SparseMatrix<double,Eigen::ColMajor,long int> SpMatrixC;
typedef Eigen::SparseMatrix<double,Eigen::RowMajor,long int> SpMatrixR;
typedef SpMatrixC::InnerIterator InIter;
typedef SpMatrixR::InnerIterator InIterR;

struct CGGMOptions {
		CGGMOptions() : 
	verbose(1),
	max_outer_iters(10),
	max_Theta_iters(1),
	max_Lambda_iters(1),
    max_ls_iters(10),
    sparse_Theta_cd(true),
    alpha(1),
    beta(0.5),
    sigma(1.0e-4),
    tol(1.0e-2),
    cd_tol(0.05) {}

	// Whether to print info messages
	int verbose;

	// Maximum iterations 
	int max_outer_iters;
	int max_Theta_iters;
	int max_Lambda_iters;
	int max_ls_iters;

    // Sparse Theta CD option for EM
    bool sparse_Theta_cd;

	// Line search parameters
	double alpha;
	double beta;
	double sigma;

	// Tolerance parameters 
	double tol;
	double cd_tol;
};

void m_step_yz(
        const Eigen::MatrixXd& Szz, // (r, r)
        const Eigen::MatrixXd& Syz, // (q, r)
        const Eigen::MatrixXd& Syy, // (q, q)
        double lambda_z,
        double lambda_y,
        SpMatrixC& Lambda,
        SpMatrixC& Theta,
        CGGMOptions& options,
        CGGMStats* stats);
#endif
