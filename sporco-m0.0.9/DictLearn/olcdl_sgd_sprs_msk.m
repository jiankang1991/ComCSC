function [D, optinf] = olcdl_sgd_sprs_msk(D0, S, lambda, W, opt)

% olcdl_sgd_sprs_msk -- Masked Online Convolutional Dictionary Learning
%         (masked SGD approach, sparse matrix computation)
%
%         argmin_{x_m,d_m} (1/2) \sum_k ||W \sum_m d_m * x_k,m - s_k||_2^2 +
%                           lambda \sum_k \sum_m ||x_k,m||_1
%
%         The solution is computed using SGD with sparse matrices
%         (see liu-2018-first).
%
% Usage:
%       [D, optinf] = olcdl_sgd_sprs_msk(D0, S, lambda, W, opt);
%
% Input:
%       D0          Initial dictionary
%       S           Input images
%       lambda      Regularization parameter
%       W           The masking matrix
%       opt         Options/algorithm parameters structure (see below)
%
% Output:
%       D           Dictionary filter set (3D array)
%       optinf      Details of optimisation
%
%
% Options structure fields:
%   Verbose          Flag determining whether iteration status is displayed.
%                    Fields are iteration number, the difference of
%                    the current dictionary with the last one, and
%                    the image index.
%   MaxMainIter      Maximum main iterations
%   eta_a            The "a" in "eta = a / (b + t)" in the paper.
%   eta_b            The "b" in "eta = a / (b + t)" in the paper.
%   DictFilterSizes  Array of size 2 x M where each column specifies the
%                    filter size (rows x columns) of the corresponding
%                    dictionary filter
%   ZeroMean         Force learned dictionary entries to be zero-mean
%
%
% Author: Jialin Liu <danny19921123@gmail.com>
%         Brendt Wohlberg <brendt@lanl.gov>
%
% This file is part of the SPORCO library. Details of the copyright
% and user license can be found in the 'License' file distributed with
% the library.

if nargin < 5,
  opt = [];
end
opt = defaultopts(opt);

eta_a = opt.eta_a;
eta_b = opt.eta_b;

% Set up status display for verbose operation
hstr = 'Itn   s(D) ';
nsep = 84;
if opt.Verbose && opt.MaxMainIter > 0,
  disp(hstr);
  disp(char('-' * ones(1,nsep)));
end

Nimg = size(S,3);

if size(S,3) > 1,
  S = reshape(S, [size(S,1) size(S,2) 1 size(S,3)]);
end

% Mean removal and normalisation projections
Pzmn = @(x) bsxfun(@minus, x, mean(mean(x,1),2));
Pnrm = @(x) bsxfun(@rdivide, x, max(sqrt(sum(sum(x.^2, 1), 2)),1));

% Start timer
tstart = tic;

% output info
optinf = struct('itstat', [], 'opt', opt);

% Initialise main working variables
D = Pnrm(D0);
G = D;
Gprv = G;

% Options for sparse coding
lambda_sc = lambda;
opt_sc = [];
opt_sc.Verbose = 0;
opt_sc.MaxMainIter = 500;
opt_sc.AutoRho = 0;
opt_sc.AutoRhoPeriod = 1;
opt_sc.RelaxParam = 1.8;
opt_sc.RelStopTol = 1e-3;
opt_sc.rho = 3;

% Use random shuffle order to read images.
epochs = (opt.MaxMainIter) / Nimg;
indices = [];
for ee = 1:(epochs+1)
        indices = [indices, randperm(Nimg)];
end

%% Main loop
k = 1;
while k <= opt.MaxMainIter,

  index = indices(k);

  % sparse coding with masking function
  SampleS = S(:,:,:,index);
  SampleW = W(:,:,index);
  opt_sc.W = SampleW;
  [X, optinf_sc] = cbpdnms(G, SampleS, lambda_sc, opt_sc);

  % computing the learning rate
  eta = eta_a / (k + eta_b);

  gra = nabla_sparse(X, G, SampleS, SampleW); % gradient of f_{mask}(d)
  G = G - eta * gra;
  G = Pnrm(G);

  sd = norm(vec(Gprv - G)); % successive differences
  g = norm(gra(:));
  Gprv = G;

  % Record and display iteration details
  tk = toc(tstart);
  optinf.itstat = [optinf.itstat; [k sd g eta tk]];
  if opt.Verbose,
    fprintf('k: %4d  sd: %.4e  index: %3d\n', k, sd, index);
  end

  k = k + 1;

end

D = G;

%% Record run time and working variables
optinf.runtime = toc(tstart);
optinf.G = G;
optinf.lambda = lambda;
optinf.lastIndex = index;
optinf.indices = indices;

if opt.Verbose && opt.MaxMainIter > 0,
  disp(char('-' * ones(1,nsep)));
end

return


function u = vec(v)

  u = v(:);

return


function g = nabla_sparse(x,D,S,W)

X = x2XsparseW(double(x), size(D), double(W));

d = reshape(double(D),[numel(D), 1]);

res = X * d - reshape(double(S),[numel(S),1]);

g = X' * res;

g = reshape(full(g),size(D));

if class(S) == 'single',
    g = single(g);
end

return


function [X] = x2XsparseW(x, dsz, W) % transfer vector x to a sparse operator X

xsz = size(x);
M = xsz(3);
x = double(x); % MATLAB only supports double sparse matrix

xx = []; % sparse x_m
ww = sparse(W);
for m = 1:M
    xx = [xx, {sparse(x(:,:,m))}];
end

% triple array form of X operator
triple = zeros(nnz(x)*dsz(1)*dsz(2),3);

num = 0;
for m = 1:M
        for dj = 1:dsz(2)
        for di = 1:dsz(1)
            index = di + (dj-1) * dsz(1) + (m-1) * dsz(1) * dsz(2);
            xx_m = reshape(ww .* circshift(xx{m},[di-1,dj-1]),[xsz(1)*xsz(2) 1]);
            [ii,~,ss] = find(xx_m);
            num_now = size(ii,1);
            triple(num+1:num+num_now,:) = [ii, repmat(index,[num_now 1]), ss];
            num = num + num_now;
        end
        end
end

triple = triple(1:num,:);

X = sparse(triple(:,1),triple(:,2),triple(:,3),xsz(1)*xsz(2),dsz(1)*dsz(2)*M);

return


function opt = defaultopts(opt)

  if ~isfield(opt,'Verbose'),
    opt.Verbose = 1;
  end
  if ~isfield(opt,'MaxMainIter'),
    opt.MaxMainIter = 200;
  end
  if ~isfield(opt,'eta_a'),
    opt.eta_a = 10;
  end
  if ~isfield(opt,'eta_b'),
    opt.eta_b = 5;
  end
  if ~isfield(opt,'ZeroMean'),
    opt.ZeroMean = 0;
  end

return
