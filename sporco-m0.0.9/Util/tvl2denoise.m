function [X, optinf] = tvl2denoise(S, lambda, opt)

% tvl2denoise -- ℓ2 Total Variation Denoising
%
%         argmin_x (1/2)||W(x - s)||_2^2 +
%                  lambda || sqrt((G_r x).^2 + (G_c x).^2) ||_2^2
%
%         The solution is computed using an ADMM approach (see
%         goldstein-2009-split and boyd-2010-distributed).
%
% Usage:
%       [X, optinf] = tvl2denoise(S, lambda, opt);
%
% Input:
%       S           Input image
%       lambda      Regularization parameter (TV)
%       opt         Algorithm parameters structure
%
% Output:
%       X           Solution array
%       optinf      Details of optimisation
%
%
% Options structure fields:
%   Verbose          Flag determining whether iteration status is displayed.
%                    Fields are iteration number, functional value,
%                    data fidelity term, l1 regularisation term, and
%                    primal and dual residuals (see Sec. 3.3 of
%                    boyd-2010-distributed). The value of rho is also
%                    displayed if options request that it is automatically
%                    adjusted.
%   MaxMainIter      Maximum main iterations
%   AbsStopTol       Absolute convergence tolerance (see Sec. 3.3.1 of
%                    boyd-2010-distributed)
%   RelStopTol       Relative convergence tolerance (see Sec. 3.3.1 of
%                    boyd-2010-distributed)
%   DatFidWeight     Data fidelity term weighting matrix
%   Yr0              Initial value for Yr
%   Yc0              Initial value for Yc
%   Ur0              Initial value for Ur
%   Uc0              Initial value for Uc
%   rho              Augmented Lagrangian parameter
%   AutoRho          Flag determining whether rho is automatically
%                    updated (see Sec. 3.4.1 of boyd-2010-distributed)
%   AutoRhoPeriod    Iteration period on which rho is updated
%   RhoRsdlRatio     Primal/dual residual ratio in rho update test
%   RhoScaling       Multiplier applied to rho when updated
%   AutoRhoScaling   Flag determining whether RhoScaling value is
%                    adaptively determined (see wohlberg-2015-adaptive). If
%                    enabled, RhoScaling specifies a maximum allowed
%                    multiplier instead of a fixed multiplier.
%   RhoRsdlTarget    Residual ratio targeted by auto rho update policy.
%   StdResiduals     Flag determining whether standard residual definitions
%                    (see Sec 3.3 of boyd-2010-distributed) are used instead
%                    of normalised residuals (see wohlberg-2015-adaptive)
%
%
% Author: Brendt Wohlberg <brendt@lanl.gov>
%
% This file is part of the SPORCO library. Details of the copyright
% and user license can be found in the 'Copyright' and 'License' files
% distributed with the library.

if nargin < 3,
  opt = [];
end
checkopt(opt, defaultopts([]));
opt = defaultopts(opt);

% Set up status display for verbose operation
hstr = ['Itn   Fnc       DFid      TV        R         S      '];
sfms = '%4d %9.2e %9.2e %9.2e %9.2e %9.2e';
nsep = 54;
if opt.AutoRho,
  hstr = [hstr '   rho   '];
  sfms = [sfms ' %9.2e'];
  nsep = nsep + 10;
end
if opt.Verbose && opt.MaxMainIter > 0,
  disp(hstr);
  disp(char('-' * ones(1,nsep)));
end


% Start timer
tstart = tic;

% Set up algorithm parameters and initialise variables
rho = opt.rho;
if isempty(rho), rho = 2*lambda + 0.1; end;
if isempty(opt.RhoRsdlTarget), opt.RhoRsdlTarget = 1; end
xsz = size(S);
Nx = prod(xsz);
Ns = Nx;
Nyv = (xsz(1)-1)*xsz(2)*size(S,3);
Nyh = xsz(1)*(xsz(2)-1)*size(S,3);
optinf = struct('itstat', [], 'opt', opt);
r = Inf; s = Inf;
epri = 0; edua = 0;
W = opt.DatFidWeight;
W2 = W.^2;
Cgs = 4*ones(size(S,1), size(S,2));
Cgs(1,:) = Cgs(1,:) - 1;
Cgs(end,:) = Cgs(end,:) - 1;
Cgs(:,1) = Cgs(:,1) - 1;
Cgs(:,end) = Cgs(:,end) - 1;

% Initialise main working variables
X = S;
if isempty(opt.Yv0),
  Yv = Dv(S);
else
  Yv = opt.Yv0;
end
Yvprv = Yv;
if isempty(opt.Uv0),
  Uv = zeros(xsz);
else
  Uv = opt.Uv0;
end
if isempty(opt.Yh0),
  Yh = Dh(S);
else
  Yh = opt.Yh0;
end
Yhprv = Yh;
if isempty(opt.Uh0),
  Uh = zeros(xsz);
else
  Uh = opt.Uh0;
end


% Main loop
k = 1;
while k <= opt.MaxMainIter && (r > epri | s > edua),

  % Solve X subproblem
  gsrrs = inf; ngs = 0;
  while gsrrs > opt.GSTol && ngs < opt.GSMaxIter,
    X = l2gsstep(X, S, Yv, Uv, Yh, Uh, rho, Cgs, W2);
    gsrrs = rrs(rho*DvT(Dv(X)) + rho*DhT(Dh(X)) + W2.*X, ...
                W2.*S + rho*DvT(Yv - Uv) + rho*DhT(Yh - Uh));
    ngs = ngs + 1;
  end

  % Solve Yv, Yh subproblem
  [Yv, Yh] = shrinktv(Dv(X) + Uv, Dh(X) + Uh, (lambda/rho));

  % Update dual variables
  Uv = Uv + Dv(X) - Yv;
  Uh = Uh + Dh(X) - Yh;

  % Compute data fidelity term in Fourier domain (note normalisation)
  Jdf = sum(vec((W.*(X-S)).^2))/2;
  Jtv = sum(vec(sqrt(Dv(X).^2 + Dh(X).^2)));
  Jfn = Jdf + lambda*Jtv;

  rvn = max(norm(vec(Dv(X))), norm(vec(Yv)));
  rhn = max(norm(vec(Dh(X))), norm(vec(Yh)));
  svn = rho*norm(vec((DvT(Uv))));
  shn = rho*norm(vec((DhT(Uh))));
  if opt.StdResiduals,
    % See pp. 19-20 of boyd-2010-distributed
    rv = norm(vec(Dv(X) - Yv));
    rh = norm(vec(Dh(X) - Yh));
    sv = norm(rho*vec(DvT(Yvprv - Yv)));
    sh = norm(rho*vec(DhT(Yhprv - Yh)));
    epriv = sqrt(Nyv)*opt.AbsStopTol + rvn*opt.RelStopTol;
    eprih = sqrt(Nyh)*opt.AbsStopTol + rhn*opt.RelStopTol;
    eduav = sqrt(Nx)*opt.AbsStopTol + svn*opt.RelStopTol;
    eduah = sqrt(Nx)*opt.AbsStopTol + shn*opt.RelStopTol;
    r = 0.5*(rv + rh); s = 0.5*(sv + sh);
    epri = 0.5*(epriv + eduah); edua = 0.5*(eduav + eduah);
  else
    % See wohlberg-2015-adaptive
    rv = norm(vec(Dv(X) - Yv)) / rvn;
    rh = norm(vec(Dh(X) - Yh)) / rhn;
    sv = norm(rho*vec(DvT(Yvprv - Yv))) / svn;
    sh = norm(rho*vec(DhT(Yhprv - Yh))) / shn;
    epriv = sqrt(Nyv)*opt.AbsStopTol/rvn + opt.RelStopTol;
    eprih = sqrt(Nyh)*opt.AbsStopTol/rhn + opt.RelStopTol;
    eduav = sqrt(Nx)*opt.AbsStopTol/svn + opt.RelStopTol;
    eduah = sqrt(Nx)*opt.AbsStopTol/shn + opt.RelStopTol;
    r = 0.5*(rv + rh); s = 0.5*(sv + sh);
    epri = 0.5*(epriv + eduah); edua = 0.5*(eduav + eduah);
  end


  % Record and display iteration details
  optinf.itstat = [optinf.itstat; [k Jfn Jdf Jtv r s epri edua rho gsrrs ngs]];
  if opt.Verbose,
    dvc = [k Jfn Jdf Jtv r s];
    if opt.AutoRho,
      dvc = [dvc rho];
    end
    disp(sprintf(sfms, dvc));
  end

  % See wohlberg-2015-adaptive and pp. 20-21 of boyd-2010-distributed
  rsf = 1;
  if opt.AutoRho,
    if k ~= 1 && mod(k, opt.AutoRhoPeriod) == 0,
      if opt.AutoRhoScaling,
        rhomlt = sqrt(r/(s*opt.RhoRsdlTarget));
        if rhomlt < 1, rhomlt = 1/rhomlt; end
        if rhomlt > opt.RhoScaling, rhomlt = opt.RhoScaling; end
      else
        rhomlt = opt.RhoScaling;
      end
      rsf = 1;
      if r > opt.RhoRsdlTarget*opt.RhoRsdlRatio*s, rsf = rhomlt; end
      if s > (opt.RhoRsdlRatio/opt.RhoRsdlTarget)*r, rsf = 1/rhomlt; end
      rho = rsf*rho;
      Uv = Uv/rsf;
      Uh = Uh/rsf;
    end
  end

  Yvprv = Yv;
  Yhprv = Yh;
  k = k + 1;

end

% Record run time and working variables
optinf.runtime = toc(tstart);
optinf.Yv = Yv;
optinf.Yh = Yh;
optinf.Uv = Uv;
optinf.Uh = Uh;
optinf.lambda = lambda;
optinf.rho = rho;

if opt.Verbose && opt.MaxMainIter > 0,
  disp(char('-' * ones(1,nsep)));
end

return


function u = vec(v)

  u = v(:);

return



function [u, v] = shrinktv(x, y, lambda)

  a = sqrt(x.^2 + y.^2);
  if isscalar(lambda),
    b = max(0, a - lambda);
  else
    b = max(0, bsxfun(@minus, a, lambda));
  end

  b(a == 0) = 0;
  a(a == 0) = 1;
  b = b./a;

  u = bsxfun(@times, b, x);
  v = bsxfun(@times, b, y);

return



function y = Dv(x)

  y = [x(2:end,:,:) - x(1:(end-1),:,:); zeros(1,size(x,2),size(x,3))];

return


function y = DvT(x)

  y = [zeros(1,size(x,2),size(x,3)); x(1:(end-1),:,:)] - ...
      [x(1:(end-1),:,:); zeros(1,size(x,2),size(x,3))];

return


function y = Dh(x)

  y = [x(:,2:end,:) - x(:,1:(end-1),:), zeros(size(x,1),1,size(x,3))];

return


function y = DhT(x)

  y = [zeros(size(x,1),1,size(x,3)), x(:,1:(end-1),:)] - ...
      [x(:,1:(end-1),:), zeros(size(x,1),1,size(x,3))];

return



function Y = l2gsstep(X, S, Yv, Uv, Yh, Uh, rho, Cgs, W2)

  X01 = [zeros(1,size(X,2),size(X,3)); X(1:(end-1),:,:)];
  X21 = [X(2:end,:,:); zeros(1,size(X,2),size(X,3))];
  X10 = [zeros(size(X,1),1,size(X,3)), X(:,1:(end-1),:)];
  X12 = [X(:,2:end,:), zeros(size(X,1),1,size(X,3))];
  Y = bsxfun(@rdivide, W2.*S + rho*(X01 + X21 + X10 + X12 + ...
             DvT(Yv - Uv) + DhT(Yh - Uh)), (bsxfun(@plus, W2, rho*Cgs)));

return




function opt = defaultopts(opt)

  if ~isfield(opt,'Verbose'),
    opt.Verbose = 0;
  end
  if ~isfield(opt,'MaxMainIter'),
    opt.MaxMainIter = 1000;
  end
  if ~isfield(opt,'AbsStopTol'),
    opt.AbsStopTol = 0;
  end
  if ~isfield(opt,'RelStopTol'),
    opt.RelStopTol = 1e-4;
  end
  if ~isfield(opt,'DatFidWeight'),
    opt.DatFidWeight = 1;
  end
  if ~isfield(opt,'GSTol'),
    opt.GSTol = 1e-4;
  end
  if ~isfield(opt,'GSMaxIter'),
    opt.GSMaxIter = 2;
  end
  if ~isfield(opt,'Yv0'),
    opt.Yv0 = [];
  end
  if ~isfield(opt,'Uv0'),
    opt.Uv0 = [];
  end
  if ~isfield(opt,'Yh0'),
    opt.Yh0 = [];
  end
  if ~isfield(opt,'Uh0'),
    opt.Uh0 = [];
  end
  if ~isfield(opt,'rho'),
    opt.rho = [];
  end
  if ~isfield(opt,'AutoRho'),
    opt.AutoRho = 1;
  end
  if ~isfield(opt,'AutoRhoPeriod'),
    opt.AutoRhoPeriod = 1;
  end
  if ~isfield(opt,'RhoRsdlRatio'),
    opt.RhoRsdlRatio = 10;
  end
  if ~isfield(opt,'RhoScaling'),
    opt.RhoScaling = 2;
  end
  if ~isfield(opt,'AutoRhoScaling'),
    opt.AutoRhoScaling = 0;
  end
  if ~isfield(opt,'RhoRsdlTarget'),
    opt.RhoRsdlTarget = [];
  end
  if ~isfield(opt,'StdResiduals'),
    opt.StdResiduals = 0;
  end

return
