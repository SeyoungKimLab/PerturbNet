function [lambdaLambdas, lambdaThetas] = regularization_suggestions(Y, X, varargin)
    numSuggestions = 10;
    avgDegreeLambda = 10;
    avgDegreeTheta = 10;
    sampleDims = 5000;
    if ~isempty(varargin)
        for i = 1:size(varargin, 2) - 1
            if strcmp(varargin{i}, 'numSuggestions')
                numSuggestions = varargin{i+1};
            elseif strcmp(varargin{i}, 'avgDegreeLambda')
                avgDegreeLambda = varargin{i+1};
            elseif strcmp(varargin{i}, 'avgDegreeTheta')
                avgDegreeTheta = varargin{i+1};
            elseif strcmp(varargin{i}, 'sampleDims')
                sampleDims = varargin{i+1};
            end
        end
    end

    Q = size(Y, 2);
    q = min(Q, sampleDims);
    Yrandperm = randperm(Q);
    Ysub_ixs = Yrandperm(1:q);
    Ysub = Y(:, Ysub_ixs);
    YsubShrinkage = q*q / (Q*Q);

    P = size(X, 2);
    p = min(P, sampleDims);
    Xrandperm = randperm(P);
    Xsub_ixs = Xrandperm(1:p);
    Xsub = X(:, Xsub_ixs);
    XsubShrinkage = p*q / (P*Q);

    q = size(Ysub, 2);
    p = size(Xsub, 2);
    n = size(Ysub, 1);
    rankingSyy = floor(max(1,min(q*q,avgDegreeLambda*q*YsubShrinkage)));
    rankingSxy = floor(max(1,min(p*q,avgDegreeTheta*q*XsubShrinkage)));
    Yc = bsxfun(@minus, Ysub, mean(Ysub));
    Xc = bsxfun(@minus, Xsub, mean(Xsub));
    Syy_vals = abs(Yc' * Yc) / n;
    Syy_vals(1:q+1:q^2) = 0;
    Syy_vals = Syy_vals(:);
    Sxy_vals = abs(Xc' * Yc) / n;
    Sxy_vals = Sxy_vals(:);
    lambdaLambdaMax = max(Syy_vals);
    lambdaThetaMax = max(Sxy_vals);
    Syy_sorted = sort(Syy_vals, 'descend');
    lambdaLambdaMin = Syy_sorted(rankingSyy);
    Sxy_sorted = sort(Sxy_vals, 'descend');
    lambdaThetaMin = Sxy_sorted(rankingSxy);
    lambdaLambdasL = log10(logspace( ...
        lambdaLambdaMin, lambdaLambdaMax*0.99, numSuggestions)); 
    lambdaLambdas = fliplr(round(lambdaLambdasL, 4));
    lambdaThetasL = log10(logspace( ...
        lambdaThetaMin, lambdaThetaMax*0.99, numSuggestions)); 
    lambdaThetas = fliplr(round(lambdaThetasL, 4));
end
