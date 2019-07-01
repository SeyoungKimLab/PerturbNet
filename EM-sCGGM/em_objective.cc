#include "em_objective.h"
#include "em_util.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::SimplicialLLT;
using Eigen::Success;
using Eigen::Upper;
using Eigen::Lower;

using namespace std;


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
		SpMatrixC& Theta_xy) {
	double n = Z.rows();
	double n_o = Y_o.rows();
	double n_h = n - n_o;
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
	double ll_xy_tr = traceProduct(Lambda_yy_sym, Y_XB, Y_XB);
	
	/*
	MatrixXd Theta_xy_dense = Theta_xy;
	MatrixXd Lambda_yy_sym_dense = Lambda_yy_sym;
	MatrixXd Sigma_y = Lambda_yy_sym_dense.inverse();
	MatrixXd XB = 	-1.0*X_o*Theta_xy_dense*Sigma_y;
	MatrixXd Y_XB_2 = Y_o - XB;
	MatrixXd Product = Y_XB_2 * Lambda_yy_sym_dense * Y_XB_2.transpose();
	double ll_xy_tr_2 = Product.trace();
	*/

	double ll_xy_logdet = -1*n_o*logdet(chol_yy.matrixL(), true);
	double ll_xy = ll_xy_tr + ll_xy_logdet;
	double ll_yz_tr = traceProduct(Lambda_zz_sym, Z_YB, Z_YB);
	double ll_yz_logdet = -1*n_o*logdet(chol_zz.matrixL(), true);
	double ll_yz = ll_yz_tr + ll_yz_logdet;
	printf("  EMObjective computed ll_xy and ll_yz \n");
	fflush(stdout);

	MatrixXd XhThT = (-1*X_h*Theta_xy).transpose();   // n_h x q
	MatrixXd eY_h = chol_yy.solve(XhThT).transpose(); // n_h x q
	MatrixXd YhThT = (-1*eY_h*Theta_yz).transpose();  // n_h x r
	MatrixXd eZX_h = chol_zz.solve(YhThT).transpose(); // n_h x r
	printf("  EMObjective computed z_h|x_h \n");
	fflush(stdout);

	MatrixXd Z_Xmu_h = Z_h - eZX_h; // n_h x r
	MatrixXd Ir = MatrixXd::Identity(r, r);
	MatrixXd Sig_z = chol_zz.solve(Ir);
	MatrixXd Sig_zx;

// *** Math to avoid inverting Lambda_y ***
// Sig_zx = inv(Lambda_z) + Beta_yz^T * inv(Lambda_y) * Beta_yz
//=inv(Lambda_z)+inv(Lambda_z)*Theta_yz^T*inv(Lambda_y)*Theta_yz*inv(Lambda_z)
//=[I_r + inv(Lambda_z)*Theta_yz^T*inv(Lambda_y)*Theta_yz]*inv(Lambda_z)
// ****************************************

	//Sig_zx = Sig_z + Sig_z * Theta_yz.transpose() 
	//	* chol_yy.solve(MatrixXd::Identity(q,q)) * Theta_yz * Sig_z;  
	//Sig_zx = Sig_z + Sig_z * Theta_yz.transpose() 
	//	* chol_yy.solve(MatrixXd(Theta_yz)) * Sig_z;
	Sig_zx = (Ir + Sig_z*(
		Theta_yz.transpose()*chol_yy.solve(Theta_yz)))*Sig_z;
	Sig_zx = Sig_zx.selfadjointView<Eigen::Upper>();
	printf("  EMObjective computed Sig_zx \n");
	fflush(stdout);

	double logdetSig_zx;
	MatrixXd iSig_zx;
	
	// PartialPivLU - both inverse and determinant
	//Eigen::PartialPivLU<MatrixXd> Sig_zx_LU(Sig_zx);
	//iSig_zx = Sig_zx_LU.solve(Ir);
	//logdetSig_zx = log(Sig_zx_LU.determinant());
	
	// LLT (Cholesky) for inverse and determinant
	Eigen::LLT<MatrixXd> Sig_zx_LLT(Sig_zx);
	iSig_zx = Sig_zx_LLT.solve(Ir);
	MatrixXd L = Sig_zx_LLT.matrixL();
	logdetSig_zx = 2*L.diagonal().array().log().sum();
	printf("  EMObjective computed iSig_zx and logdetSig_zx \n");
	fflush(stdout);

	MatrixXd innerZ_Xmu_h = Z_Xmu_h.transpose()*Z_Xmu_h;
	double ll_xz = traceProduct(iSig_zx, innerZ_Xmu_h) + n_h*logdetSig_zx;
	double pen = lambda_zz*L1NormOffDiag(Lambda_zz_sym) + 
		lambda_yz*L1Norm(Theta_yz) + lambda_yy*L1NormOffDiag(Lambda_yy_sym) +
		lambda_xy*L1Norm(Theta_xy);
	double f = (ll_xz + ll_xy + ll_yz)/n + pen;
	return f;
}


