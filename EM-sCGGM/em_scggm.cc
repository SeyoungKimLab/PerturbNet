//#define EIGEN_USE_LAPACKE
//#define EIGEN_USE_BLAS

#include "em_scggm.h"
#include "m_step_xy.h"
#include "m_step_yz.h"
#include "em_util.h"

#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>
#include <stdint.h>
#include <sys/time.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::SimplicialLLT;
using Eigen::Success;
using Eigen::Upper;

using namespace std;

/*
double logdetEigenDense(const Eigen::MatrixXd& A) {
}
*/

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
	/*
	Eigen::LLT<MatrixXd> Sig_zx_decomp(Sig_zx);
	MatrixXd iSig_zx = Sig_zx_decomp.solve(Ir);
	*/
	MatrixXd innerZ_Xmu_h = Z_Xmu_h.transpose()*Z_Xmu_h;
	double ll_xz = traceProduct(iSig_zx, innerZ_Xmu_h) +
		n_h*log(Sig_zx_decomp.determinant());
	/*
	double ll_xz = traceProduct(iSig_zx, innerZ_Xmu_h) +
		n_h*log(Sig_zx.determinant());
	*/
	double pen = lambda_zz*L1NormOffDiag(Lambda_zz_sym) + 
		lambda_yz*L1Norm(Theta_yz) + lambda_yy*L1NormOffDiag(Lambda_yy_sym) +
		lambda_xy*L1Norm(Theta_xy);
	return (ll_xz + ll_xy + ll_yz)/n + pen;
}


// Lambda_zz and Lambda_yy are upper diag, not symmetric
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
		EMStats* stats) {

	srand(1);
	//Eigen::setNbThreads(4);
	long n = X.rows();
	long n_o = Y_o.rows();
	long n_h = n - n_o;
	long p = Theta_xy.rows();
	long q = Lambda_yy.rows();
	long r = Lambda_zz.rows();
	struct timeval start_time, current_time;
	gettimeofday(&start_time, NULL);
	
	MatrixXd X_h = X.bottomRows(n_h);
	MatrixXd Z_h = Z.bottomRows(n_h);
	MatrixXd Szz = (1.0/n) * Z.transpose() * Z;
	MatrixXd Iq = MatrixXd::Identity(q, q);
	MatrixXd Ir = MatrixXd::Identity(r, r);

	MatrixXd eY_haha = Y_o;
	MatrixXd X_haha = X.topRows(n_o);
	MatrixXd Z_haha = Z.topRows(n_o);
	MatrixXd Szz_haha = Z_haha.transpose() * Z_haha;
	MatrixXd Syz_haha = Y_o.transpose() * Z_haha;
	MatrixXd Syy_haha = Y_o.transpose() * Y_o;

	vector< vector<double> > X_vec(p, vector<double>(X.rows(), 0));
	for (long i = 0; i < p; i++) {
		for (long nn = 0; nn < X.rows(); nn++) {
			X_vec[i][nn] = X(nn, i);
		}
	}

	struct timeval tic_time, toc_time;
	double obj_0 = EMObjective(Z, Y_o, X, 
		lambda_zz, lambda_yz, lambda_yy, lambda_xy,
		Lambda_zz, Theta_yz, Lambda_yy, Theta_xy);
	stats->objval.push_back(obj_0);
	if (!options.quiet) {
		printf("EM-sCGGM f_0=%f\n", obj_0);
	}
	double obj = obj_0;

	HugeOptions options_xy;
	options_xy.quiet = options.quiet;
	options_xy.max_outer_iters = options.max_outer_iters;
	options_xy.sigma = options.sigma;
	options_xy.tol = options.tol;
	options_xy.num_blocks_Lambda = options.num_blocks_Lambda;
	options_xy.num_blocks_Theta = options.num_blocks_Theta;
	options_xy.memory_usage = options.memory_usage;

	for (int em_iter = 0; em_iter < options.max_em_iters; ++em_iter) {
		// E-step
		gettimeofday(&tic_time, NULL);
		SpMatrixC Lambda_zz_sym;
		Lambda_zz_sym = Lambda_zz.selfadjointView<Upper>();
		SpMatrixC Lambda_yy_sym;
		Lambda_yy_sym = Lambda_yy.selfadjointView<Upper>();
		SimplicialLLT<SpMatrixC> chol_zz(Lambda_zz_sym);
		SpMatrixC Theta_zy = Theta_yz.transpose();
		MatrixXd nZ_hThzy_X_hThxy = -1*(Z_h*Theta_zy + X_h*Theta_xy);
	
		MatrixXd Sig_y_giv_xz;
		if (options.sparse_E_step) {
			// Avoid dense inverse
			//     Convert from Eigen format
			smat_t Lambda_yy_alt;
			vector<Triplet> Lambda_yy_alt_triplets;
			for (long j = 0; j < q; j++) {
				for (InIter it(Lambda_yy_sym, j); it; ++it) {
					Lambda_yy_alt_triplets.push_back(
						Triplet(it.row(), it.col(), it.value()));
				}
			}
			Lambda_yy_alt.setTriplets(q, Lambda_yy_alt_triplets);
			//     Compute tall matrix V = Theta_yz*(chol_zz^-T)
			//MatrixXd Lzz = chol_zz.permutationPinv() * chol_zz.matrixL();
			//MatrixXd V_eigen = Theta_yz * Lzz.inverse().transpose();
			MatrixXd Sigma_zz = chol_zz.solve(Ir);
			MatrixXd V_eigen = Theta_yz * Sigma_zz;
			
			vector< vector<double> > V(q, vector<double>(r, 0));
			for (long qq = 0; qq < q; qq++) {
				for (long rr = 0; rr < r; rr++) {
					V[qq][rr] = V_eigen(qq,rr);
				}
			}
			//    Compute inverse via CG method
			Sig_y_giv_xz = MatrixXd::Identity(q, q);
			vector<double> Sig_y_giv_xz_i(q, 0);
			for (long i = 0; i < q; i++) {
				Lambda_yy_alt.ComputeInvAVVt(V, i, Sig_y_giv_xz_i, 
					options_xy.grad_tol);
				for (long j = 0; j < q; j++) {
					Sig_y_giv_xz(i,j) = Sig_y_giv_xz_i[j];
				}
			}
		} else {
			// Performs dense inverse
			printf("  E-step dense inverse \n");
			fflush(stdout);
			MatrixXd Inv_y_giv_xz = Theta_yz * chol_zz.solve(Theta_zy);
			Inv_y_giv_xz += Lambda_yy_sym;
			Eigen::LLT<MatrixXd> chol_Inv_y_giv_xz(Inv_y_giv_xz);
			Sig_y_giv_xz = chol_Inv_y_giv_xz.solve(Iq);
			Sig_y_giv_xz = 0.5*(Sig_y_giv_xz + Sig_y_giv_xz.transpose());
		}

		// Compute sufficient statistics
		MatrixXd mu_y_giv_xz = nZ_hThzy_X_hThxy*Sig_y_giv_xz;
		MatrixXd eY(n, q);
    	eY << Y_o,
			  mu_y_giv_xz;
		MatrixXd eYtY = n_h*Sig_y_giv_xz + eY.transpose()*eY;
		MatrixXd Syy = (0.5/n) * (eYtY + eYtY.transpose());
		MatrixXd eYtZ = eY.transpose() * Z;
		MatrixXd Syz = (1.0/n) * eYtZ;

		// Convert sufficient statistics to vector format
		vector< vector<double> > Syy_vv(q, vector<double>(q, 0));
		for (long i = 0; i < q; i++) {
			for (long j = 0; j < q; j++) {
				Syy_vv[i][j] = Syy(i,j);
			}
		}
		vector< vector<double> > eY_vec(q, vector<double>(eY.rows(), 0));
		for (long i = 0; i < q; i++) {
			for (long nn = 0; nn < eY.rows(); nn++) {
				eY_vec[i][nn] = eY(nn,i);
			}
		}

		gettimeofday(&toc_time, NULL);
		stats->time_e_step.push_back(toddiff(&tic_time, &toc_time));
		if (!options.quiet) {
			printf("EM iter %i, done with E step \n", em_iter);
			fflush(stdout);
		}

		// M-step
		CGGMStats stats_yz;
		HugeStats stats_xy;
		for (long m_step_iter = 0; m_step_iter < 1; m_step_iter++) {
			// Convert parameters from Eigen format
			smat_t Lambda_yy_smat;
			sparse_t Theta_xy_sparse;
			
			// Eigen Lambda has i<=j while smat Lambda has i>=j
			Lambda_yy_sym = Lambda_yy.selfadjointView<Upper>();

			vector<Triplet> Lambda_yy_triplets;
			for (long j = 0; j < q; j++) {
				for (InIter it(Lambda_yy_sym, j); it; ++it) {
					Lambda_yy_triplets.push_back(
						Triplet(it.row(), it.col(), it.value()));
				}
			}
			Lambda_yy_smat.setTriplets(q, Lambda_yy_triplets);
			vector<Triplet> Theta_xy_triplets;
			for (long j = 0; j < q; j++) {
				for (InIter it(Theta_xy, j); it; ++it) {
					Theta_xy_triplets.push_back(
						Triplet(it.row(), it.col(), it.value()));
				}
			}
			Theta_xy_sparse.setTriplets(p, q, Theta_xy_triplets);

			// M-step YZ
			m_step_yz(Szz, Syz, Syy, 
				lambda_zz, lambda_yz, Lambda_zz, Theta_yz,
				options, &stats_yz);
			double obj_after_yz = EMObjective(Z, Y_o, X, 
				lambda_zz, lambda_yz, lambda_yy, lambda_xy,
				Lambda_zz, Theta_yz, Lambda_yy, Theta_xy);
			if (!options.quiet) {
				printf("EM iter %i, after yz obj=%f \n", em_iter, obj_after_yz);
				if (obj_after_yz > obj) {
					printf("WARNING: EM obj increased after YZ!\n");
				}
			}
			obj = obj_after_yz;

			// M-step XY
			m_step_xy(Syy_vv, eY_vec, X_vec, 
				lambda_yy, lambda_xy, Lambda_yy_smat, Theta_xy_sparse,
				options_xy, stats_xy);
		
			// Convert parameters to Eigen format
			vector<TripE> Lambda_yy_tripEs;
			for (long i = 0; i < q; i++) {
				for (long idx = Lambda_yy_smat.row_ptr[i]; 
					idx < Lambda_yy_smat.row_ptr[i+1]; idx++) {
					// Swap row and col, because:
					//  smat.cpp: if (triplets[n].col > curr_row) { is_symmetric = 1; }
					//  em_evaluate.cc: if (i <= j) { triplets.push_back(TripE(i-1, j-1, val)); }
					long smat_row = i;
					long smat_col = Lambda_yy_smat.col_idx[idx];
					long eigen_row = smat_col;
					long eigen_col = smat_row;
					Lambda_yy_tripEs.push_back(
						TripE(eigen_row, eigen_col, 
						Lambda_yy_smat.values[idx]));
				}
			}
			Lambda_yy.setFromTriplets(Lambda_yy_tripEs.begin(),
				Lambda_yy_tripEs.end());
			vector<TripE> Theta_xy_tripEs;
			for (long i = 0; i < p; i++) {
				for (long idx = Theta_xy_sparse.row_ptr[i]; 
					idx < Theta_xy_sparse.row_ptr[i+1]; idx++) {
					Theta_xy_tripEs.push_back(
						TripE(i, Theta_xy_sparse.col_idx[idx], 
						Theta_xy_sparse.values[idx]));
				}
			}
			Theta_xy.setFromTriplets(Theta_xy_tripEs.begin(),
				Theta_xy_tripEs.end());
			if (!options.quiet) {
				printf("Converted params to Eigen format \n");
			}

			double obj_after_xy = EMObjective(Z, Y_o, X, 
				lambda_zz, lambda_yz, lambda_yy, lambda_xy,
				Lambda_zz, Theta_yz, Lambda_yy, Theta_xy);
			if (!options.quiet) {
				printf("EM iter %i, after xy obj=%f \n", em_iter, obj_after_xy);
				if (obj_after_xy > obj) {
					printf("WARNING: EM obj increased after XY!\n");
				}
				fflush(stdout);
			}
			obj = obj_after_xy;
		}
		
		// Save stats
		stats->objval.push_back(obj);
		gettimeofday(&current_time, NULL);
		stats->time.push_back(toddiff(&start_time, &current_time));
		stats->l1norm.push_back(
			L1Norm(Lambda_zz)+L1Norm(Theta_yz)+L1Norm(Lambda_yy)+L1Norm(Theta_xy));
		stats->iters_yz.push_back(stats_yz.time.size());
		stats->iters_xy.push_back(stats_xy.time.size());
		stats->active_lambda_zz.push_back(Lambda_zz.nonZeros()); //assume upper
		stats->active_theta_yz.push_back(Theta_yz.nonZeros());
		stats->active_lambda_yy.push_back(Lambda_yy.nonZeros()); //assume upper
		stats->active_theta_xy.push_back(Theta_xy.nonZeros());
		stats->time_lambda_zz_active.push_back(
			accumulate(stats_yz.time_lambda_active.begin(), 
			stats_yz.time_lambda_active.end(), 0.0));
		stats->time_lambda_zz_cd.push_back(
			accumulate(stats_yz.time_lambda_cd.begin(), 
			stats_yz.time_lambda_cd.end(), 0.0));
		stats->time_theta_yz_active.push_back(
			accumulate(stats_yz.time_theta_active.begin(), 
			stats_yz.time_theta_active.end(), 0.0));
		stats->time_theta_yz_cd.push_back(
			accumulate(stats_yz.time_theta_cd.begin(), 
			stats_yz.time_theta_cd.end(), 0.0));
		stats->time_lambda_yy_active.push_back(
			accumulate(stats_xy.time_lambda_active.begin(), 
			stats_xy.time_lambda_active.end(), 0.0));
		stats->time_lambda_yy_cd.push_back(
			accumulate(stats_xy.time_lambda_cd.begin(), 
			stats_xy.time_lambda_cd.end(), 0.0));
		stats->time_theta_xy_active.push_back(
			accumulate(stats_xy.time_theta_active.begin(), 
			stats_xy.time_theta_active.end(), 0.0));
		stats->time_theta_xy_cd.push_back(
			accumulate(stats_xy.time_theta_cd.begin(), 
			stats_xy.time_theta_cd.end(), 0.0));

		int min_emiters = 3; // minimum 1 
		int rel_emiters = 2; // minimum 1 would be previous objective
		if (em_iter > min_emiters) {
			double old_obj = stats->objval[stats->objval.size()-1-rel_emiters];
			double relchange = abs((obj-old_obj)/old_obj)/rel_emiters;
			if (relchange < options.em_tol) {
				printf("Stopping EM at iter %i, relchange=%f \n", 
					em_iter, relchange);
				break;
			}
		}
	} // em iteration 
	
}

