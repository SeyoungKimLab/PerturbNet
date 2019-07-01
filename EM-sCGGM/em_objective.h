#ifndef _EM_OBJECTIVE_H_
#define _EM_OBJECTIVE_H_

#include <Eigen/Dense>
#include <Eigen/Sparse>

typedef Eigen::SparseMatrix<double,Eigen::ColMajor,long int> SpMatrixC;

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
		SpMatrixC& Lambda_yy, // still works if Lambda_yy_sym
		SpMatrixC& Theta_xy);

#endif
