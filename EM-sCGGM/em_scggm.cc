//#define EIGEN_USE_LAPACKE
//#define EIGEN_USE_BLAS

#include "em_scggm.h"
#include "m_step_xy.h"
#include "m_step_yz.h"
#include "em_util.h"
#include "em_objective.h"

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
	int num_threads = 1;
#ifdef _OPENMP
	num_threads = min(options.max_threads, omp_get_max_threads());
	omp_set_num_threads(num_threads);
	omp_set_dynamic(true);
	//Eigen::setNbThreads(num_threads);
#endif
	if (options.verbose >= 2) {
		printf("num threads:%d \n", num_threads);
		fflush(stdout);
	}

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
	MatrixXd Iq;
	MatrixXd Ir = MatrixXd::Identity(r, r);
	if (!options.sparse_E_step) {
		Iq = MatrixXd::Identity(q, q);
	}
	MatrixXd Y_oTY_o = Y_o.transpose() * Y_o;

	vector< vector<double> > X_vec(p, vector<double>(n, 0.0));
	for (long i = 0; i < p; i++) {
		for (long nn = 0; nn < n; nn++) {
			X_vec[i][nn] = X(nn, i);
		}
	}

	struct timeval tic_time, toc_time;
	double obj_0 = EMObjective(Z, Y_o, X, 
		lambda_zz, lambda_yz, lambda_yy, lambda_xy,
		Lambda_zz, Theta_yz, Lambda_yy, Theta_xy);
	stats->objval.push_back(obj_0);
	if (options.verbose >= 1) {
		printf("EM-sCGGM f_0=%f\n", obj_0);
		fflush(stdout);
	}
	double obj = obj_0;

	HugeOptions options_xy;
	options_xy.verbose = options.verbose;
	options_xy.max_outer_iters = 1;
	options_xy.sigma = options.sigma;
	options_xy.tol = options.tol;
	options_xy.num_blocks_Lambda = options.num_blocks_Lambda;
	options_xy.num_blocks_Theta = options.num_blocks_Theta;
	options_xy.memory_usage = options.memory_usage;

	CGGMOptions options_yz;
	options_yz.verbose = options.verbose;
	options_yz.max_outer_iters = 1;
	options_yz.sparse_Theta_cd = options.sparse_Theta_cd;
	options_yz.sigma = options.sigma;
	options_yz.tol = options.tol;

	// Matrices computed by E-step
	MatrixXd eY(n, q);
	MatrixXd Syy(q, q);
	MatrixXd Syz(q, r);

	for (int em_iter = 0; em_iter < options.max_em_iters; ++em_iter) {
		// E-step
// The problem is probably here since happens regardless of sparse_E_step
		gettimeofday(&tic_time, NULL);
		SpMatrixC Lambda_zz_sym;
		Lambda_zz_sym = Lambda_zz.selfadjointView<Upper>();
		SpMatrixC Lambda_yy_sym;
		Lambda_yy_sym = Lambda_yy.selfadjointView<Upper>();
		SimplicialLLT<SpMatrixC> chol_zz(Lambda_zz_sym);
		SpMatrixC Theta_zy = Theta_yz.transpose();
		MatrixXd nZ_hThzy_X_hThxy = -1*(Z_h*Theta_zy + X_h*Theta_xy);
	
		MatrixXd Sig_y_giv_xz(q, q);
		printf("E-step prep \n");
		fflush(stdout);
		if (options.sparse_E_step) {
			// Avoid dense inverse
			//     Convert from Eigen format
			printf("beginning sparse E-step \n");
			fflush(stdout);
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
			MatrixXd Sigma_zz = chol_zz.solve(Ir);
			MatrixXd V_eigen = Theta_yz * Sigma_zz;
			
			vector< vector<double> > V(q, vector<double>(r, 0));
			for (long qq = 0; qq < q; qq++) {
				for (long rr = 0; rr < r; rr++) {
					V[qq][rr] = V_eigen(qq,rr);
				}
			}
			printf("computed V \n");
			fflush(stdout);
			//    Compute inverse via CG method
			vector<double> Sig_y_giv_xz_i(q, 0);
			for (long i = 0; i < q; i++) {
				Lambda_yy_alt.ComputeInvAVVt(V, i, Sig_y_giv_xz_i, 
					options_xy.grad_tol);
				for (long j = 0; j < q; j++) {
					Sig_y_giv_xz(i,j) = Sig_y_giv_xz_i[j];
				}
			}
			printf("computed Sig_y_giv_xz \n");
			fflush(stdout);
		} else {
			// Performs dense inverse
			printf("beginning dense E-step \n");
			fflush(stdout);
			MatrixXd Theta_zy_dense = Theta_zy;
			MatrixXd Theta_yz_dense = Theta_yz;
			MatrixXd Sig_z_Theta_zy = chol_zz.solve(Theta_zy_dense);
			MatrixXd Inv_y_giv_xz = Theta_yz_dense * Sig_z_Theta_zy;
			Inv_y_giv_xz += Lambda_yy_sym;
			Eigen::LLT<MatrixXd> chol_Inv_y_giv_xz(Inv_y_giv_xz);
			Sig_y_giv_xz = chol_Inv_y_giv_xz.solve(Iq);
			Sig_y_giv_xz = Sig_y_giv_xz.selfadjointView<Eigen::Upper>();
		}

		// Compute sufficient statistics
		MatrixXd eY_h = nZ_hThzy_X_hThxy*Sig_y_giv_xz;
		eY.topRows(n_o) = Y_o;
		eY.bottomRows(n_h) = eY_h;
		printf("computed eY \n");
		fflush(stdout);
		double nd = static_cast<double>(n);
		//MatrixXd eYTeY = (Y_oTY_o + eY_h.transpose()*eY_h) / nd;
		MatrixXd eYTeY = (eY.transpose() * eY) / nd;
		printf("computed eY'eY \n");
		fflush(stdout);
		Syy = (static_cast<double>(n_h)/nd)*Sig_y_giv_xz + eYTeY;
		printf("computed Syy \n");
		fflush(stdout);
		Syy = Syy.selfadjointView<Eigen::Upper>();
		printf("sym Syy \n");
		fflush(stdout);
		Syz = (1.0/nd) * eY.transpose() * Z;
		printf("computed Syz \n");
		fflush(stdout);

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
		if (options.verbose >= 1) {
			printf("EM iter %i, done with E step \n", em_iter);
			fflush(stdout);
		}

		// M-step
		CGGMStats stats_yz;
		HugeStats stats_xy;
		for (long m_step_iter = 0; m_step_iter < options.max_M_iters; m_step_iter++) {
			// Convert parameters from Eigen format
			smat_t Lambda_yy_smat;
			sparse_t Theta_xy_sparse;
			
			// Eigen Lambda has i<=j while smat Lambda has i>=j
			Lambda_yy_sym = Lambda_yy.selfadjointView<Upper>();

			vector<Triplet> Lambda_yy_triplets;
			//printf("\t\t Converting from Eigen \n");
			for (long j = 0; j < q; j++) {
				for (InIter it(Lambda_yy_sym, j); it; ++it) {
					/*
					if (it.row() == it.col()) {
						printf("(%ld,%ld)=%f \n", it.row(), it.col(), it.value());
					}
					*/
					if (it.row() >= it.col()) {	
						//printf("Smat_t (%ld,%ld)=%f \n", it.row(), it.col(), it.value());

						Lambda_yy_triplets.push_back(
							Triplet(it.row(), it.col(), it.value()));
					}
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
				options_yz, &stats_yz);
			double obj_after_yz = EMObjective(Z, Y_o, X, 
				lambda_zz, lambda_yz, lambda_yy, lambda_xy,
				Lambda_zz, Theta_yz, Lambda_yy, Theta_xy);
			if (options.verbose >= 1) {
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
					if (eigen_row <= eigen_col) {
						//printf("Eigen (%ld,%ld)=%f \n", eigen_row, eigen_col,
						//	Lambda_yy_smat.values[idx]);
						Lambda_yy_tripEs.push_back(
							TripE(eigen_row, eigen_col, 
								Lambda_yy_smat.values[idx]));
					}
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

			double obj_after_xy = EMObjective(Z, Y_o, X, 
				lambda_zz, lambda_yz, lambda_yy, lambda_xy,
				Lambda_zz, Theta_yz, Lambda_yy, Theta_xy);
			if (options.verbose >= 1) {
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

