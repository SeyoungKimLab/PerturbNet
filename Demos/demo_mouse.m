addpath('../EM-sCGGM');

Z = dlmread('../Examples/mouse_pheno_406_278.txt');
Y_o = dlmread('../Examples/mouse_expr_381_1306.txt');
X = dlmread('../Examples/mouse_geno_406_363.txt');

lambdaLambda_z = 0.8;
lambdaTheta_yz = 0.5;
lambdaLambda_y = 0.8;
lambdaTheta_xy = 0.5;
options.verbose = 1;
[mLambda_z, mTheta_yz, mLambda_y, mTheta_xy, mStats] = em_scggm( ...
    Z, Y_o, X, ...
    lambdaLambda_z, lambdaTheta_yz, lambdaLambda_y, lambdaTheta_xy, ...
    options);

