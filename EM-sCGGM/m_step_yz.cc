//#define EIGEN_USE_LAPACKE
//#define EIGEN_USE_BLAS

#include "m_step_yz.h"
#include "em_util.h"

#include <cstdlib>
#include <iostream>
#include <vector>
#include <stdint.h>
#include <sys/time.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

typedef Eigen::Triplet<double> TripE;

using Eigen::LDLT;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::SimplicialLLT;
using Eigen::Success;
using Eigen::Upper;

using namespace std;

struct TimeThetaCD {
        TimeThetaCD() :
    cd(0), qr(0) {}
    double cd;
    double qr;
};

struct TimeLambdaCD {
        TimeLambdaCD() :
    cd(0), ls(0) {}
    double cd;
    double ls; // includes update to R
};

struct LambdaState {
	LambdaState(
		const double logdetLambda_,
		const double trRtQ_)
		: logdetLambda(logdetLambda_),
	      trRtQ(trRtQ_) {}
	double logdetLambda;
	double trRtQ; // tr(Sigma*Theta'*Sxx*Theta)
};


// Objective function to minimize
// Interacts with Lambda as symmetric, not upper diag
// Fast version
double Objective_yz(
		const MatrixXd& Szz,
		const MatrixXd& Syz,
		const MatrixXd& Syy,
		double lambda_z,
		double lambda_y,
		const SpMatrixC& Lambda,
		const SpMatrixC& Theta,
		const MatrixXd& Gamma) {
	// f = -logdet(Lambda) + tr(Syy*Lambda + 2Sxy'*Theta + R'*Q)
	/*
    printf("-logdetLambda:%f trLambdaSzz:%f 2trThetaSyz:%f trThetaGamma:%f\n",
        -logdet(Lambda), traceProduct(Lambda, Szz), 2*traceProduct(Theta,Syz),
        traceProduct(Theta, Gamma));
	*/
        
	return -logdet(Lambda) + traceProduct(Lambda, Szz) +
		2.0*traceProduct(Theta, Syz) + traceProduct(Theta, Gamma) +
		lambda_y*L1Norm(Theta) + lambda_z*L1NormOffDiag(Lambda);
}	

// Returns subgradient
// Adds 0s to Theta at locations of free set
double ThetaActiveSet_yz(
		const MatrixXd& Syz,
		const MatrixXd& Gamma,
		SpMatrixC& Theta,
		double lambda) {
	long q = Syz.rows();
	long r = Syz.cols();
	double subgrad = 0.0;
	MatrixXd G;
	G = 2.0*Syz + 2.0*Gamma;

	vector<TripE> triplets;
	vector<TripE> tripletsZeros;
	triplets.reserve(Theta.nonZeros());
	
	for (long j = 0; j < r; j++) {
		InIter it(Theta,j);
		for (long i = 0; i < q; i++) {
			double Theta_ij = 0;
			if (it && it.row() == i) {
				Theta_ij = it.value();
				++it;
			}
			if (Theta_ij != 0 || fabs(G(i,j)) > lambda) {
				triplets.push_back(TripE(i, j, Theta_ij));
				subgrad += fabs(L1SubGrad(Theta_ij, G(i,j), lambda));
			}
		}
	}
	Theta.setFromTriplets(triplets.begin(), triplets.end());
	return subgrad;
}

// Returns subgradient
// Interacts with Lambda as upper diag (i <= j)
// Adds 0s to Lambda at locations of free set
double LambdaActiveSet_yz(
		const MatrixXd& Szz,
		SpMatrixC& Lambda,
		const MatrixXd& Sigma,
		const MatrixXd& Psi,
		double lambda) {
	long r = Lambda.cols();
	double subgrad = 0.0;
	
	vector<TripE> triplets;
	vector<TripE> tripletsZeros;
	triplets.reserve(Lambda.nonZeros());
	MatrixXd G = Szz - Sigma - Psi;
	
	for (long j = 0; j < r; j++) {
		InIter it(Lambda, j);

		for (long i = 0; i < j; i++) {
			double Lambda_ij = 0;
			if (it && it.row() == i) {
				Lambda_ij = it.value();
				++it;
			}
			if (Lambda_ij != 0 || fabs(G(i,j)) > lambda) {
				triplets.push_back(TripE(i, j, Lambda_ij));
				subgrad += 2*fabs(L1SubGrad(Lambda_ij, G(i,j), lambda));
			}
		}
		triplets.push_back(TripE(j, j, it.value()));
		subgrad += fabs(G(j,j));
	}
	Lambda.setFromTriplets(triplets.begin(), triplets.end());
	return subgrad;
}

// Coordinate descent for Theta
// Uses Sigma
// Updates Theta, SyyTheta, Gamma, and Psi
void ThetaCoordinateDescent_yz(
		const MatrixXd& Szz, // debugging
		const MatrixXd& Syz, 
		const MatrixXd& Syy,
		SpMatrixC& Theta,
		MatrixXd& Sigma,
		MatrixXd& SyyTheta,
		MatrixXd& Gamma,
        MatrixXd& Psi,
		double lambda,
		const CGGMOptions& options,
		TimeThetaCD& time_report_theta) {
	struct timeval start_time, end_time;
	gettimeofday(&start_time, NULL);

    long q = Theta.rows();
	long r = Theta.cols();
	MatrixXd V;
	if (options.sparse_Theta_cd) {
		V = Theta * Sigma;
	} else {
		MatrixXd ThetaDense = MatrixXd(Theta);
		V = ThetaDense * Sigma;
	}
	vector<long> columns;
	columns.reserve(r);
	for (long j = 0; j < r; ++j) {
		columns.push_back(j);
	}

	for (int iter = 0; iter < options.max_Theta_iters; ++iter) {
		// Shuffles columns
		for (long ix1 = 0; ix1 < r; ++ix1) {
			long ix2 = ix1 + rand() % (r - ix1);
			long tmp = columns[ix1];
			columns[ix1] = columns[ix2];
			columns[ix2] = tmp;
		}

		for (long j_ix = 0; j_ix < r; ++j_ix) {
			long j = columns[j_ix];
			long nnz_j = Theta.col(j).nonZeros();

			// Shuffles within columns
			vector<InIter*> itThetas;
			itThetas.reserve(nnz_j);
			InIter itTheta(Theta, j);
			for (; itTheta; ++itTheta) {
				itThetas.push_back(new InIter(itTheta));
			}
			for (long ix1 = 0; ix1 < nnz_j; ++ix1) {
				long ix2 = ix1 + rand() % (nnz_j - ix1);
				InIter* tmp = itThetas[ix1];
				itThetas[ix1] = itThetas[ix2];
				itThetas[ix2] = tmp;
			}

			// Performs coordinate descent
			for (long ix = 0; ix < nnz_j; ++ix) {
				InIter* itPtr = itThetas[ix];
				long i = itPtr->row();
				double Theta_ij = itPtr->value();

				double a = 2*Sigma(j,j)*Syy(i,i);
				//printf("Sigma_jj:%f Syy_ii:%f \n", Sigma(j,j), Syy(i,i));
				double b = 2*Syz(i,j) + 2*Syy.col(i).dot(V.col(j));
				double c = Theta_ij;
				double mu = -c + SoftThreshold(c - b/a, lambda/a);
				itPtr->valueRef() += mu;
				V.row(i) += mu*Sigma.row(j);
			}

			for (long ix = 0; ix < nnz_j; ++ix) {
				delete itThetas.back();
				itThetas.pop_back();
			}
		}
	}
	gettimeofday(&end_time, NULL);
	double time_theta_cd = toddiff(&start_time, &end_time);
	time_report_theta.cd += time_theta_cd;

	gettimeofday(&start_time, NULL);
	if (options.sparse_Theta_cd) {
		SyyTheta = Syy * Theta;
		Gamma = SyyTheta * Sigma;
		MatrixXd SigmaThetaT = Sigma * Theta.transpose();
		Psi = SigmaThetaT * Gamma;
	} else {
		MatrixXd ThetaDense = MatrixXd(Theta);
		SyyTheta = Syy * ThetaDense;
		Gamma = SyyTheta * Sigma;
		Psi = Sigma * (Theta.transpose() * Gamma);
	}
	gettimeofday(&end_time, NULL);
	double time_theta_qr = toddiff(&start_time, &end_time);
	time_report_theta.qr += time_theta_qr;
}

// Coordinate descent for Lambda
// Interacts with Lambda as upper diag (i <= j)
// Uses
// Updates Lambda, Sigma, and Gamma
void LambdaCoordinateDescent_yz(
		const MatrixXd& Szz,
		const MatrixXd& Syz, // debugging
		const MatrixXd& Syy, // debugging
		SpMatrixC& Lambda,
		const SpMatrixC& Theta,
		MatrixXd& Sigma,
		const MatrixXd& SyyTheta,
		MatrixXd& Gamma,
		MatrixXd& Psi,
		double lambda_z,
		double lambda_y,
		const CGGMOptions& options,
		TimeLambdaCD& time_report_lambda) {
	
	struct timeval start_time, end_time;
	gettimeofday(&start_time, NULL);
    //printf("yahey\n");
    //printf("within LambdaCD: Gamma_00:%f \n", Gamma(0,0));
    //printf("within LambdaCD: Psi_00:%f \n", Psi(0,0));
	
	// Sets up constants
	long r = Lambda.rows();
	MatrixXd Ir = MatrixXd::Identity(r, r);
	
	// Sets up updated matrices
	SpMatrixC Delta = Lambda; // all 0s
	for (long k = 0; k < Delta.outerSize(); ++k) {
		for (InIter it(Delta,k); it; ++it) {
			it.valueRef() = 0.0;
		}
	}
	MatrixXd U;
	U = Delta * Sigma;

	// Columns to shuffle
	vector<long> columns;
	columns.reserve(r);
	for (long j = 0; j < r; ++j) {
		columns.push_back(j);
	}

	for (int iter = 0; iter < options.max_Lambda_iters; ++iter) {
		// Shuffles columns
		for (long ix1 = 0; ix1 < r; ++ix1) {
			long ix2 = ix1 + rand() % (r - ix1);
			long tmp = columns[ix1];
			columns[ix1] = columns[ix2];
			columns[ix2] = tmp;
		}
		for (long j_ix = 0; j_ix < r; ++j_ix) {
			long j = columns[j_ix];
			long nnz_j = Lambda.col(j).nonZeros();

			// Shuffles within columns
			vector<InIter*> itLambdas;
			vector<InIter*> itDeltas;
			itLambdas.reserve(nnz_j);
			itDeltas.reserve(nnz_j);
			InIter itLambda(Lambda, j);
			InIter itDelta(Delta, j);
			for (; itLambda && itDelta; ++itLambda, ++itDelta) {
				itLambdas.push_back(new InIter(itLambda));
				itDeltas.push_back(new InIter(itDelta));
			}
			for (long ix1 = 0; ix1 < nnz_j; ++ix1) {
				long ix2 = ix1 + rand() % (nnz_j - ix1);
				InIter* tmp = itLambdas[ix1];
				itLambdas[ix1] = itLambdas[ix2];
				itLambdas[ix2] = tmp;
				tmp = itDeltas[ix1];
				itDeltas[ix1] = itDeltas[ix2];
				itDeltas[ix2] = tmp;
			}

			// Performs coordinate descent
			for (long ix = 0; ix < nnz_j; ++ix) {
				InIter* itLambdaPtr = itLambdas[ix];
				InIter* itDeltaPtr = itDeltas[ix];
				long i = itLambdaPtr->row();
				double Lambda_ij = itLambdaPtr->value();
				double Delta_ij = itDeltaPtr->value();
				
				if (i == j) { // diagonal elements
					double a = Sigma(i,i)*Sigma(i,i) + 2*Sigma(i,i)*Psi(i,i);
					double b = Szz(i,i) - Sigma(i,i) - Psi(i,i) 
						+ Sigma.row(i)*U.col(i) + 2*Psi.row(i)*U.col(i);
					double mu = -b/a;
					itDeltaPtr->valueRef() += mu;
					U.row(i) += mu*Sigma.row(i);
				} else { // off-diagonal elements
					double a = Sigma(i,j)*Sigma(i,j) + Sigma(i,i)*Sigma(j,j) 
						+ Sigma(i,i)*Psi(j,j) + 2*Sigma(i,j)*Psi(i,j) 
						+ Sigma(j,j)*Psi(i,i);
					double b = Szz(i,j) - Sigma(i,j) + Sigma.row(i)*U.col(j)
						- Psi(i,j) + Psi.row(i)*U.col(j) + Psi.row(j)*U.col(i);
					double c = Lambda_ij + Delta_ij;  
					double mu = -c + SoftThreshold(c - b/a, lambda_z/a);
					itDeltaPtr->valueRef() += mu;
					U.row(i) += mu*Sigma.row(j);
					U.row(j) += mu*Sigma.row(i);
				}
			}

			for (long ix = 0; ix < nnz_j; ++ix) {
				delete itLambdas.back();
				delete itDeltas.back();
				itLambdas.pop_back();
				itDeltas.pop_back();
			}
		}
	}
	gettimeofday(&end_time, NULL);
	time_report_lambda.cd += toddiff(&start_time, &end_time);
	
	// Backtracking line search
	gettimeofday(&start_time, NULL);
	double alpha = options.alpha;
	bool success = false;
	SpMatrixC Lambda_sym;
	Lambda_sym = Lambda.selfadjointView<Upper>();
	SpMatrixC Delta_sym;
	Delta_sym = Delta.selfadjointView<Upper>();
	SpMatrixC Lambda_alpha; // symmetric, not upper diag
	double logdetLambda_alpha;

	double trGradDelta = traceProduct(Delta, Szz) 
		- traceProduct(Delta, Sigma) - traceProduct(Delta, Psi);
	SpMatrixC LambdaPlusDelta = Lambda_sym + Delta_sym;
	double RHS = options.sigma*(trGradDelta 
		+ lambda_z*(L1NormOffDiag(LambdaPlusDelta) - L1NormOffDiag(Lambda)));
	double obj_noalpha = Objective_yz(Szz, Syz, Syy, lambda_z, 
		lambda_y, Lambda, Theta, Gamma);

	for (int lsiter = 0; lsiter < options.max_ls_iters; ++lsiter) {
		Lambda_alpha = Lambda_sym + alpha*Delta_sym;
		SimplicialLLT<SpMatrixC> cholesky(Lambda_alpha);
        MatrixXd Sigma_alpha = cholesky.solve(Ir);
		if (cholesky.info() != Success) {
			if (options.verbose >= 1) {
				printf("      line search %d, alpha=%f, not PD\n", lsiter, alpha);
				fflush(stdout);
			}
			alpha *= options.beta;
			continue;
		}
		logdetLambda_alpha = logdet(cholesky.matrixL(), 0);
		MatrixXd Gamma_alpha = SyyTheta * Sigma_alpha;
		double LHS = Objective_yz(Szz, Syz, Syy, lambda_z, lambda_y, 
			Lambda_alpha, Theta, Gamma_alpha) - obj_noalpha;

		if (LHS <= alpha*RHS) {
			success = true;
			Lambda = Lambda + alpha*Delta;
			Sigma = Sigma_alpha;
			Gamma = Gamma_alpha;
		    Psi = Sigma * (Theta.transpose() * Gamma);
            /*
			if (!options.quiet) {
				printf(
                    "   line search %d, alpha=%f, sufficient decrease=%f\n",
				    lsiter, alpha, LHS);
			}
            */
			break;
		} else if (options.verbose >= 1) {
			printf("      line search %d, alpha=%f, insufficient decrease=%6.4g\n",
				lsiter, alpha, LHS);
			fflush(stdout);
		}
		alpha *= options.beta;
	}
	if (success) {
	} else if (options.verbose >= 1) {
		printf("     Lambda line search failed\n");
	}
	gettimeofday(&end_time, NULL);
	time_report_lambda.ls += toddiff(&start_time, &end_time);
}

void m_step_yz(
        const Eigen::MatrixXd& Szz, // (r, r)
        const Eigen::MatrixXd& Syz, // (q, r)
        const Eigen::MatrixXd& Syy, // (q, q)
        double lambda_z,
        double lambda_y,
        SpMatrixC& Lambda,
        SpMatrixC& Theta,
        CGGMOptions& options,
        CGGMStats* stats) {

	srand(1);
	long q = Theta.rows();
	long r = Theta.cols();

	struct timeval start_time, current_time;
	gettimeofday(&start_time, NULL);

	SpMatrixC Lambda_sym;
	Lambda_sym = Lambda.selfadjointView<Upper>();
	SimplicialLLT<SpMatrixC> cholesky(Lambda_sym);
	if (cholesky.info() != Success) { 
		if (options.verbose >= 1) {
			printf("   YZ Lambda0 not positive definite\n");
		}
		return;
	}
	MatrixXd Sigma;
	Sigma = cholesky.solve(MatrixXd::Identity(r,r));
    MatrixXd SyyTheta;
	SyyTheta = Syy * Theta;
	MatrixXd Gamma;
    Gamma = SyyTheta * Sigma;
    MatrixXd Psi;
	struct timeval mini_start_time, mini_end_time;

	double f_0 = Objective_yz(
		Szz, Syz, Syy, lambda_z, lambda_y, 
		Lambda_sym, Theta, Gamma);	
	stats->objval.push_back(f_0);
	if (options.verbose >= 2) {
		printf("   M-step-YZ f_0 = %f\n", f_0);
		fflush(stdout);
	}
	for (int t_outer = 0; t_outer < options.max_outer_iters; ++t_outer) {
		Psi = Sigma * (Theta.transpose() * Gamma);
		// Calculate the subgradient and free sets
		gettimeofday(&mini_start_time, NULL);
    	double subgrad_theta = ThetaActiveSet_yz(
			Syz, Gamma, Theta, lambda_y);
		gettimeofday(&mini_end_time, NULL);
		double theta_active_time = toddiff(&mini_start_time, &mini_end_time);
		if (options.verbose >= 2) {
			printf("   YZ iteration %d: finished ThetaActive  %ld \n", 
				t_outer, Theta.nonZeros());
			fflush(stdout);
		}

		gettimeofday(&mini_start_time, NULL);
		double subgrad_lambda = LambdaActiveSet_yz(
			Szz, Lambda, Sigma, Psi, lambda_z);
		gettimeofday(&mini_end_time, NULL);
		double lambda_active_time = toddiff(&mini_start_time, &mini_end_time);
		if (options.verbose >= 2) {
			printf("   YZ iteration %d: finished LambdaActive %ld \n",
				t_outer, Lambda.nonZeros());
			fflush(stdout);
		}

    	double l1_norm_theta = L1Norm(Theta);
		double l1_norm_lambda = L1NormUpperDiag(Lambda);
		
		double subgrad = subgrad_theta + subgrad_lambda;
		double l1_norm = l1_norm_theta + l1_norm_lambda;
		if (t_outer > 0) {
			stats->subgrad.push_back(subgrad);
			stats->l1norm.push_back(l1_norm);
		}

    	if (subgrad < options.tol*l1_norm) {
			if (options.verbose >= 1) {
				printf("   YZ converged, subgrad=%f, norm=%f\n", 
					subgrad, l1_norm);
				fflush(stdout);
			}
			break;
		}

		// CD for Lambda, also updates Sigma, R
		if (options.verbose >= 1) {
			Lambda_sym = Lambda.selfadjointView<Upper>();
			double objLambdaCD = Objective_yz(
				Szz, Syz, Syy, lambda_z, lambda_y, 
				Lambda_sym, Theta, Gamma);
			printf("   YZ iteration %d: starting LambdaCD, obj=%.4f \n",
				t_outer, objLambdaCD);
			fflush(stdout);
		}
		// Calculate the subgradient and free sets
		struct TimeLambdaCD time_report_lambda;
		gettimeofday(&mini_start_time, NULL);
		LambdaCoordinateDescent_yz(
			Szz, Syz, Syy, Lambda, Theta, 
			Sigma, SyyTheta, Gamma, Psi, lambda_z, lambda_y, 
			options, time_report_lambda);
		gettimeofday(&mini_end_time, NULL);
		double lambda_cd_time =	toddiff(&mini_start_time, &mini_end_time);

		// CD for Theta, also updates SyyTheta, Gamma, Psi
		if (options.verbose >= 1) {
			Lambda_sym = Lambda.selfadjointView<Upper>();
			double objThetaCD = Objective_yz(
				Szz, Syz, Syy, lambda_z, lambda_y, 
				Lambda_sym, Theta, Gamma);
			printf("   YZ iteration %d: starting ThetaCD, obj=%.4f \n",
				t_outer, objThetaCD);
			fflush(stdout);
		}
		struct TimeThetaCD time_report_theta;
		gettimeofday(&mini_start_time, NULL);
		ThetaCoordinateDescent_yz(
			Szz, Syz, Syy, Theta, Sigma, SyyTheta, Gamma, Psi, lambda_y, 
			options, time_report_theta);
		gettimeofday(&mini_end_time, NULL);
		double theta_cd_time = toddiff(&mini_start_time, &mini_end_time);

		Lambda_sym = Lambda.selfadjointView<Upper>();
		double f = Objective_yz(
			Szz, Syz, Syy, lambda_z, lambda_y, 
			Lambda_sym, Theta, Gamma);	

		if (options.verbose >= 1) {
			printf("   YZ iteration %d, Lambda(subgrad=%f,active=%ld,norm=%f\n",
	 			t_outer, subgrad_lambda, Lambda.nonZeros(), l1_norm_lambda);
			printf("                   Theta(subgrad=%f,active=%ld,norm=%f\n",
				subgrad_theta, Theta.nonZeros(), l1_norm_theta);
			printf("                   ending f=%.4f \n", f);
			fflush(stdout);
		}

		stats->objval.push_back(f);
		gettimeofday(&current_time, NULL);
		stats->time.push_back(toddiff(&start_time, &current_time));
		long active_set_size = Theta.nonZeros() + Lambda.nonZeros();
		stats->active_set_size.push_back((double)active_set_size);
		stats->active_theta.push_back((double)Theta.nonZeros());
		stats->active_lambda.push_back((double)Lambda.nonZeros());

		// Output iteration time breakdown
		stats->time_lambda_active.push_back(lambda_active_time);
		stats->time_theta_active.push_back(theta_active_time);
		stats->time_lambda_cd.push_back(lambda_cd_time);
		stats->time_lambda_cd_cd.push_back(time_report_lambda.cd);
		stats->time_lambda_cd_ls.push_back(time_report_lambda.ls);
		stats->time_theta_cd.push_back(theta_cd_time);
		stats->time_theta_cd_cd.push_back(time_report_theta.cd);
		stats->time_theta_cd_qr.push_back(time_report_theta.qr);
	} // outer Newton iteration 

	if (options.verbose >= 1) {
		printf("   M-step-YZ f* - f_0  =  %f\n", stats->objval.back()-f_0);
	}
}
