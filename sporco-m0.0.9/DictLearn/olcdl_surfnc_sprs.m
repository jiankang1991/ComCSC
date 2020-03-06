function [D, optinf] = olcdl_surfnc_sprs(D0, S, lambda, opt)

% olcdl_surfnc_sprs -- Online Convolutional Dictionary Learning
%         (surrogate function approach, sparse matrix computation)
%
%         argmin_{x_m,d_m} (1/2) \sum_k ||\sum_m d_m * x_k,m - s_k||_2^2 +
%                           lambda \sum_k \sum_m ||x_k,m||_1
%
%         The solution is computed using the surrogate function
%         approach (see liu-2017-online and liu-2018-first)
%
% Usage:
%       [D, optinf] = olcdl_surfnc_sprs(D0, S, lambda, opt);
%
% Input:
%       D0          Initial dictionary
%       S           Input images
%       lambda      Regularization parameter
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
%   tol              The stopping tolerance for the first step
%   SampleN          The size of split training images
%   p                Forgetting exponent
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

if nargin < 4,
  opt = [];
end
opt = defaultopts(opt);

SampleN = opt.SampleN;
p = opt.p;
tol = opt.tol;

% Set up status display for verbose operation
hstr = 'Itn   s(D) ';
nsep = 84;

if opt.Verbose && opt.MaxMainIter > 0,
  disp(hstr);
  disp(char('-' * ones(1,nsep)));
end

Nimg = size(S,3);

if size(S,3) > 1,
  % Insert singleton 3rd dimension (for number of filters) so that
  % 4th dimension is number of images in input s volume
  S = reshape(S, [size(S,1) size(S,2) 1 size(S,3)]);
end

% Dictionary size may be specified when learning multiscale
% dictionary
if isempty(opt.DictFilterSizes),
  dsz = [size(D0,1) size(D0,2)];
else
  dsz = opt.DictFilterSizes;
end

% Mean removal and normalisation projections
Pnrm = @(x) bsxfun(@rdivide, x, max(sqrt(sum(sum(x.^2, 1), 2)),1));

% Start timer
tstart = tic;

% output info
optinf = struct('itstat', [], 'opt', opt);

% Initialise main working variables
D = Pnrm(D0);
G = D;
Gprv = G;
approx_G = G;

% Options for sparse coding steps
lambda_sc = lambda;
opt_sc = [];
opt_sc.Verbose = 0;
opt_sc.MaxMainIter = 500;
opt_sc.AutoRho = 1;
opt_sc.AutoRhoPeriod = 1;
opt_sc.RelaxParam = 1.8;
opt_sc.RelStopTol = 1e-3;
opt_sc.rho = 10;


% A^t and b^t for on-line update
At = zeros(dsz(1)*dsz(2)*size(D0,3));
bt = zeros(dsz(1)*dsz(2)*size(D0,3), 1);
if class(S) == 'single'
  At = single(At);
  bt = single(bt);
end

% variables used in inner loop
k = 1;
alpha_sum = 0;
inner_k = 0;

% Use random shuffle order to read images.
epochs = (opt.MaxMainIter) / Nimg;
indices = [];
for ee = 1:(epochs+1)
  indices = [indices, randperm(Nimg)];
end

%% Main loop
while k <= opt.MaxMainIter,

  index = indices(k);

  Nx = floor(size(S,1)/SampleN);
  Ny = floor(size(S,2)/SampleN);
  order_x = randperm(Nx);
  order_y = randperm(Ny);

  for sp_index_x = 1:(Nx)
    for sp_index_y = 1:(Ny)

      % image splitting
      randleft = 1 + (order_x(sp_index_x)-1) * SampleN;
      randtop = 1 + (order_y(sp_index_y)-1) * SampleN;
      randright = randleft + SampleN - 1;
      randbottom = randtop + SampleN - 1;

      % Sparse coding
      SampleS = S(randleft:randright,randtop:randbottom,:,index);

      [X, ~] = cbpdn(G, SampleS, lambda_sc, opt_sc);

      if (nnz(isnan(X))>0)
        continue;
      end
      if (nnz(X)==0)
        continue;
      end

      Xop = x2Xsparse(X, dsz);
      xtx = full(Xop' * Xop); % X^t X is not sparse
      xs = full(Xop' * double(reshape(SampleS,[numel(SampleS),1])));

      % update A and b
      inner_k = inner_k + 1;
      alpha = (1 - 1/(inner_k))^p;
      alpha_sum = alpha_sum * alpha + 1;
      At = alpha * At + xtx;
      bt = alpha * bt + xs;

      % step size for FISTA
      eta = 1 / ( norm(vec(At / alpha_sum)));

      t = 1;
      InnerLoop = 1000;

      % frequency domain FISTA
      for inner_loop_d  = 1:InnerLoop

        % G is the main iterate; approx_G is the auxillary iterate.
        gra = At * reshape(approx_G, [numel(approx_G), 1]) - bt;
        gra = gra/alpha_sum;
        G = approx_G - eta * reshape(gra,size(G));
        G = Pnrm(G);

        fpr = norm(vec(approx_G - G)); % fixed point residual for FISTA

        % Nesterov acceleration
        t_next = ( 1 + sqrt(1+4*t^2) ) / 2;
        approx_G = G + (t-1)/t_next * (G - Gprv);

        Gprv = G;
        t = t_next;

        if fpr <= tol/(1+k)
          break;
        end

      end
    end
  end

  tk = toc(tstart);
  optinf.itstat = [optinf.itstat; [k inner_k inner_loop_d fpr tk]];
  if opt.Verbose,
    fprintf('k: %d split_region: %d inner_loop: %d fpr: %d\n',...
            k, inner_k, inner_loop_d, fpr);
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


function X = x2Xsparse(x, dsz) % transfer vector x to a sparse operator X

xsz = size(x);
M = xsz(3);
x = double(x); % MATLAB only supports double sparse matrix

xx = []; % sparse x_m
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
      xx_m = reshape(circshift(xx{m},[di-1,dj-1]),[xsz(1)*xsz(2) 1]);
      [ii,~,ss] = find(xx_m);
      num_now = size(ii,1);
      triple(num+1:num+num_now,:) = [ii, repmat(index,[num_now 1]), ss];
      num = num + num_now;
    end
  end
end

X = sparse(triple(:,1),triple(:,2),triple(:,3),xsz(1)*xsz(2),dsz(1)*dsz(2)*M);

return


function opt = defaultopts(opt)

  if ~isfield(opt,'Verbose'),
    opt.Verbose = 1;
  end
  if ~isfield(opt,'MaxMainIter'),
    opt.MaxMainIter = 80;
  end
  if ~isfield(opt,'tol'),
    opt.tol = 1e-2;
  end
  if ~isfield(opt,'SampleN'),
    opt.SampleN = 128;
  end
  if ~isfield(opt,'p'),
    opt.p = 10;
  end
  if ~isfield(opt,'DictFilterSizes'),
    opt.DictFilterSizes = [];
  end
  if ~isfield(opt,'ZeroMean'),
    opt.ZeroMean = 0;
  end

return
