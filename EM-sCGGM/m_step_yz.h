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

void m_step_yz(
        const Eigen::MatrixXd& Szz, // (r, r)
        const Eigen::MatrixXd& Syz, // (q, r)
        const Eigen::MatrixXd& Syy, // (q, q)
        double lambda_z,
        double lambda_y,
        SpMatrixC& Lambda,
        SpMatrixC& Theta,
        EMOptions& options,
        CGGMStats* stats);
#endif
