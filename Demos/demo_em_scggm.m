addpath('../EM-sCGGM');
rng(0);

n = 400; n_o = 100; p = 200; q = 150; r = 50;
Lambda_y = full(spdiags([0.3*ones(q,1) ones(q,1) 0.3*ones(q,1)],[-1 0 1], q, q));
p_influence = floor(p * 0.5);
Theta_xy = [full(sprand(p_influence, q, 0.01)); zeros(p-p_influence, q)];
Lambda_z = full(spdiags([0.3*ones(r,1) ones(r,1) 0.3*ones(r,1)],[-1 0 1], r, r));
q_influence = floor(q * 0.5);
Theta_yz = [full(sprand(q_influence, r, 0.01)); zeros(q-q_influence, r)];

X = randn(n, p);
meanY = -X*Theta_xy*inv(Lambda_y); 
noiseY = ((chol(Lambda_y,'lower')')\randn(q, n))';
Y = meanY + noiseY;
meanZ = -Y*Theta_yz*inv(Lambda_z); 
noiseZ = ((chol(Lambda_z,'lower')')\randn(r, n))';
Z = meanZ + noiseZ;

Y_o = Y(1:n_o,:);

lambdaLambda_z = 0.7;
lambdaTheta_yz = 1;
lambdaLambda_y = 0.3;
lambdaTheta_xy = 0.4;
lambda.xy = lambdaTheta_xy;
lambda.yy = lambdaLambda_y;
lambda.yz = lambdaTheta_yz;
lambda.zz = lambdaLambda_z;

clear options;
options.verbose = 1;
options.sparse_E_step = 0;
options.sparse_Theta_cd = 1;
options.max_M_iters = 1;
options.max_em_iters = 10;
options.em_tol = 1e-8;
[estLambda_z, estTheta_yz, estLambda_y, estTheta_xy, estStats] = em_scggm(...
    Z, Y_o, X, ...
    lambdaLambda_z, lambdaTheta_yz, lambdaLambda_y, lambdaTheta_xy, ...
    options);
estParams.xy = estTheta_xy;
estParams.yy = estLambda_y;
estParams.yz = estTheta_yz;
estParams.zz = estLambda_z;

figure('name', 'EM comparison');
subplot(3,4,4);
spy(Lambda_z); title('true \Lambda_z');
subplot(3,4,3);
spy(Theta_yz); title('true \Theta_{yz}');
subplot(3,4,2);
spy(Lambda_y); title('true \Lambda_y');
subplot(3,4,1);
spy(Theta_xy); title('true \Theta_xy');
subplot(3,4,8);
spy(estLambda_z); title('est \Lambda_z');
subplot(3,4,7);
spy(estTheta_yz); title('est \Theta_{yz}');
subplot(3,4,6);
spy(estLambda_y); title('est \Lambda_y');
subplot(3,4,5);
spy(estTheta_xy); title('est \Theta_{xy}');
