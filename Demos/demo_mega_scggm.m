addpath('../Mega-sCGGM');

% Simple example
n = 100; p = 200; q = 150;
LambdaSimple = full(spdiags([0.3*ones(q,1) ones(q,1) 0.3*ones(q,1)],[-1 0 1], q, q));
p_influence = floor(p * 0.5);
ThetaSimple = [full(sprand(p_influence, q, 0.01)); zeros(p-p_influence, q)];
X = randn(n, p); % (10 samples x 15 features)
meanY = -X*ThetaSimple*inv(LambdaSimple); 
noiseY = ((chol(LambdaSimple,'lower')')\randn(q, n))';% (10 samples x 12 features)
Y = meanY + noiseY;

lambdaLambda = 0.5;
lambdaTheta = 1;
[estLambdaSimple, estThetaSimple, statsSimple] = mega_scggm(...
    Y, X, lambdaLambda, lambdaTheta);
figure('name', 'Simple demo');
subplot(2,2,1); spy(LambdaSimple); title('true Lambda');
subplot(2,2,2); spy(ThetaSimple); title('true Theta');
subplot(2,2,3); spy(estLambdaSimple); title('est Lambda');
subplot(2,2,4); spy(estThetaSimple); title('est Theta');

% Warm-starting example
figure('name', 'Warm-starting demo');
options.max_outer_iters = 20;
numSuggestions = 5;
[lambdaLambdas, lambdaThetas] = regularization_suggestions(Y, X, ...
    'numSuggestions', numSuggestions);
[estLambda, estTheta, estStats] = mega_scggm(Y, X, lambdaLambdas(1), lambdaThetas(1), options);
subplot(numSuggestions, 2, 1); spy(estLambda);
subplot(numSuggestions, 2, 2); spy(estTheta);

for reg_ix=2:numSuggestions
    lambdaLambda = lambdaLambdas(reg_ix);
    lambdaTheta = lambdaThetas(reg_ix);
    options.Lambda0 = estLambda;
    options.Theta0 = estTheta;
    [estLambda, estTheta] = mega_scggm(Y, X, lambdaLambda, lambdaTheta, options);
    subplot(numSuggestions, 2, (reg_ix-1)*2+1); spy(estLambda);
    subplot(numSuggestions, 2, (reg_ix-1)*2+2); spy(estTheta);
end
