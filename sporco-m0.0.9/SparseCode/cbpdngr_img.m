function [Y2, optinf] = cbpdngr_img(D, S, lambda, mu, opt)
% cbpdngr -- Convolutional Basis Pursuit DeNoising with Gradient Regularization
%
%         argmin_{x_k} (1/2)||\sum_k d_k * x_k - s||_2^2 +
%                           lambda \sum_k ||x_k||_1 +
%                           (mu/2) \sum_k ||G_r \sum_k d_k * x_k||_2^2 +
%                           (mu/2) \sum_k ||G_c \sum_k d_k * x_k||_2^2
%
%         The solution is computed using an ADMM approach (see
%         boyd-2010-distributed) with efficient solution of the main
%         linear systems (see wohlberg-2016-efficient and
%         wohlberg-2016-convolutional2).
%
% Usage:
%       [Y, optinf] = cbpdngr(D, S, lambda, mu, opt);
%
% Input:
%       D           Dictionary filter set (3D array)
%       S           Input image
%       lambda      Regularization parameter (l1)
%       mu          Regularization parameter (l2 of gradient)
%       opt         Algorithm parameters structure
%
% Output:
%       Y           Dictionary coefficient map set (3D array)
%       optinf      Details of optimisation
%
%
% Options structure fields:
%   Verbose          Flag determining whether iteration status is displayed.
%                    Fields are iteration number, functional value,
%                    data fidelity term, l1 regularisation term, gradient
%                    regularisation term, and primal and dual residuals
%                    (see Sec. 3.3 of boyd-2010-distributed). The value of
%                    rho is also displayed if options request that it is
%                    automatically adjusted.
%   MaxMainIter      Maximum main iterations
%   AbsStopTol       Absolute convergence tolerance (see Sec. 3.3.1 of
%                    boyd-2010-distributed)
%   RelStopTol       Relative convergence tolerance (see Sec. 3.3.1 of
%                    boyd-2010-distributed)
%   L1Weight         Weighting array for coefficients in l1 norm of X
%   GrdWeight        Weighting array for coefficients in l2 norm of
%                    gradient of X
%   Y0               Initial value for Y
%   U0               Initial value for U
%   rho              ADMM penalty parameter
%   AutoRho          Flag determining whether rho is automatically updated
%                    (see Sec. 3.4.1 of boyd-2010-distributed)
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
%   RelaxParam       Relaxation parameter (see Sec. 3.4.3 of
%                    boyd-2010-distributed)
%   NonNegCoef       Flag indicating whether solution should be forced to
%                    be non-negative
%   NoBndryCross     Flag indicating whether all solution coefficients
%                    corresponding to filters crossing the image boundary
%                    should be forced to zero.
%   AuxVarObj        Flag determining whether objective function is computed
%                    using the auxiliary (split) variable
%   HighMemSolve     Use more memory for a slightly faster solution
%
% CONVOLUTIONAL SPARSE REPRESENTATIONS WITH GRADIENT PENALTIES
%
% This file is part of the SPORCO library. Details of the copyright
% and user license can be found in the 'License' file distributed with
% the library.

if nargin < 5,
  opt = [];
end
checkopt(opt, defaultopts([]));
opt = defaultopts(opt);

% Set up status display for verbose operation
hstr = 'Itn   Fnc       DFid      l1        Grd        r         s      ';
sfms = '%4d %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e';
nsep = 64;
if opt.AutoRho,
  hstr = [hstr '   rho  '];
  sfms = [sfms ' %9.2e'];
  nsep = nsep + 10;
end
if opt.Verbose && opt.MaxMainIter > 0,
  disp(hstr);
  disp(char('-' * ones(1,nsep)));
end
% Start timer
tstart = tic;

% Collapsing of trailing singleton dimensions greatly complicates
% handling of both SMV and MMV cases. The simplest approach would be
% if S could always be reshaped to 4d, with dimensions consisting of
% image rows, image cols, a single dimensional placeholder for number
% of filters, and number of measurements, but in the single
% measurement case the third dimension is collapsed so that the array
% is only 3d.
if size(S,3) > 1,
    xsz = [size(S,1) size(S,2) size(D,3) size(S,3)];
    hrm = [1 1 1 size(S,3)];
    % Insert singleton 3rd dimension (for number of filters) so that
    % 4th dimension is number of images in input s volume
    S = reshape(S, [size(S,1) size(S,2) 1 size(S,3)]);
else
    xsz = [size(S,1) size(S,2) size(D,3) 1]; %%% size of feature maps
    hrm = 1;
end
xrm = [1 1 size(D,3)];
% Compute filters in DFT domain
Df = fft2(D, size(S,1), size(S,2)); %%% pad to the image size when do fft2
grv = [-1 1];%%% gradient in row direction
Grf = fft2(grv, size(S,1), size(S,2));
gcv = [-1 1]';%%% gradient in column direction
Gcf = fft2(gcv, size(S,1), size(S,2));
if isscalar(opt.GrdWeight),
  opt.GrdWeight = opt.GrdWeight * ones(size(D,3), 1);
end
wgr = reshape(opt.GrdWeight, [1 1 length(opt.GrdWeight)]);

% set new conv filter as (22)-(23) in the paper
GrDf = bsxfun(@times, Df, Grf); 
GrDf = bsxfun(@times, GrDf, wgr);
GcDf = bsxfun(@times, Df, Gcf);
GcDf = bsxfun(@times, GcDf, wgr);



% Convolve-sum and its Hermitian transpose
Dop = @(x) sum(bsxfun(@times, Df, x), 3);
DHop = @(x) bsxfun(@times, conj(Df), x);

% Compute signal in DFT domain
Sf = fft2(S);
% S convolved with all filters in DFT domain
DSf = DHop(Sf);

% Default lambda is 1/10 times the lambda value beyond which the
% solution is a zero vector
if nargin < 3 | isempty(lambda),
  b = ifft2(DHop(Sf));
  lambda = 0.1*max(vec(abs(b)));
end

% Set up algorithm parameters and initialise variables
rho = opt.rho;
if isempty(rho), rho = 50*lambda+1; end;
if isempty(opt.RhoRsdlTarget),
  if opt.StdResiduals,
    opt.RhoRsdlTarget = 1;
  else
    opt.RhoRsdlTarget = 1 + (18.3).^(log10(lambda) + 1);
  end
end
Nx = prod(xsz);
optinf = struct('itstat', [], 'opt', opt);    
r = Inf;
s = Inf;
epri = 0;
edua = 0;

% Initialise main working variables
X = [];
if isempty(opt.Y0),
  Y2 = zeros(xsz, class(S));
else
  Y2 = opt.Y0;
end

Y0 = zeros([size(S,1), size(S,2), 1, size(S,3)]);
Y1 = zeros([size(S,1), size(S,2), 1, size(S,3)]);

% Y = cat(3, Y0, Y1, Y2);

Yprv = Y2;

if isempty(opt.U0),
  if isempty(opt.Y0),
    U2 = zeros(xsz, class(S));
  else
    U2 = (lambda/rho)*sign(Y2);
  end
else
  U2 = opt.U0;
end
U0 = zeros([size(S,1), size(S,2), 1, size(S,3)]);
U1 = zeros([size(S,1), size(S,2), 1, size(S,3)]);



% Main loop
k = 1;
while k <= opt.MaxMainIter && (r > epri | s > edua),
    
    Df_GrDf_GcDf = cat(4, Df, sqrt(rho) * GrDf, sqrt(rho) * GcDf);
    
    % Solve X subproblem
    Xf = solvemdbi_ism(Df_GrDf_GcDf, rho, DSf + rho*(fft2(Y2 - U2) + ...
            bsxfun(@times, conj(GrDf), fft2(Y0-U0)) + bsxfun(@times, conj(GcDf), fft2(Y1-U1))));

    X = ifft2(Xf);
    
    % See pg. 21 of boyd-2010-distributed
    if opt.RelaxParam == 1,
        Xr = X;
    else
        Xr = opt.RelaxParam*X + (1-opt.RelaxParam)*Y2;
    end
    
    % Solve Y subproblem
    
    Y2 = Shrinkcomp(Xr + U2, (lambda/rho)*opt.L1Weight);
    [Y0, Y1] = block_shrink_cpx(ifft2(GrDf.*Xf)+U0, ifft2(GcDf.*Xf)+U1, mu/rho);
    
    if opt.NonNegCoef,
        Y2(Y2 < 0) = 0;
    end
    if opt.NoBndryCross,
        Y2((end-size(D,1)+2):end,:,:,:) = 0;
        Y2(:,(end-size(D,2)+2):end,:,:) = 0;
    end
    
    % Update dual variable
    U0 = U0 + ifft2(GrDf.*Xf) - Y0;
    U1 = U1 + ifft2(GcDf.*Xf) - Y1;
    U2 = U2 + Xr - Y2;
    
    % Compute data fidelity term in Fourier domain (note normalisation)
    if opt.AuxVarObj,
        Yf = fft2(Y2); % This represents unnecessary computational cost
        Jdf = sum(vec(abs(sum(bsxfun(@times,Df,Yf),3)-Sf).^2))/(2*xsz(1)*xsz(2));
        Jl1 = sum(abs(vec(bsxfun(@times, opt.L1Weight, Y2))));
%         Jgr = sum(vec((bsxfun(@times, GfW, conj(Yf).*Yf))))/(2*xsz(1)*xsz(2));
        Jgr = sum(abs(vec(sqrt(conj(GrDf.*Yf).*(GrDf.*Yf) + ...
            conj(GcDf.*Yf).*(GcDf.*Yf)))))/(2*xsz(1)*xsz(2));
    else
        Jdf = sum(vec(abs(sum(bsxfun(@times,Df,Xf),3)-Sf).^2))/(2*xsz(1)*xsz(2));
        Jl1 = sum(abs(vec(bsxfun(@times, opt.L1Weight, X))));
%         Jgr = sum(vec((bsxfun(@times, GfW, conj(Xf).*Xf))))/(2*xsz(1)*xsz(2));
        Jgr = sum(abs(vec(sqrt(conj(GrDf.*Xf).*(GrDf.*Xf) + ...
            conj(GcDf.*Xf).*(GcDf.*Xf)))))/(2*xsz(1)*xsz(2));
    end
    Jfn = Jdf + lambda*Jl1 + mu*Jgr;

    nX = norm(X(:)); nY = norm(Y2(:)); nU = norm(U2(:));
    if opt.StdResiduals,
    % See pp. 19-20 of boyd-2010-distributed
        r = norm(vec(X - Y2));
        s = norm(vec(rho*(Yprv - Y2)));
        epri = sqrt(Nx)*opt.AbsStopTol+max(nX,nY)*opt.RelStopTol;
        edua = sqrt(Nx)*opt.AbsStopTol+rho*nU*opt.RelStopTol;
    else
        % See wohlberg-2015-adaptive
        r = norm(vec(X - Y2))/max(nX,nY);
        s = norm(vec(Yprv - Y2))/nU;
        epri = sqrt(Nx)*opt.AbsStopTol/max(nX,nY)+opt.RelStopTol;
        edua = sqrt(Nx)*opt.AbsStopTol/(rho*nU)+opt.RelStopTol;
    end
    % Record and display iteration details
    tk = toc(tstart);
    optinf.itstat = [optinf.itstat; [k Jfn Jdf Jl1 Jgr r s epri edua rho tk]];
    if opt.Verbose,
        if opt.AutoRho,
            disp(sprintf(sfms, k, Jfn, Jdf, Jl1, Jgr, r, s, rho));
        else
            disp(sprintf(sfms, k, Jfn, Jdf, Jl1, Jgr, r, s));
        end
    end
    
    % See wohlberg-2015-adaptive and pp. 20-21 of boyd-2010-distributed
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
        U2 = U2/rsf;
        end
    end
    
    Yprv = Y2;
    k = k + 1;

end

% Record run time and working variables
optinf.runtime = toc(tstart);
optinf.X = X;
optinf.Xf = Xf;
optinf.Y = Y2;
optinf.U = U2;
optinf.lambda = lambda;
optinf.mu = mu;
optinf.rho = rho;
% End status display for verbose operation
if opt.Verbose && opt.MaxMainIter > 0,
  disp(char('-' * ones(1,nsep)));
end

end

function u = vec(v)

  u = v(:);

end

function [P1, P2] = block_shrink_cpx(z1, z2, tau)
    
    re_z1 = real(z1);
    im_z1 = imag(z1);
    
    re_z2 = real(z2);
    im_z2 = imag(z2);
    
    z1_norm_2 = re_z1.^2 + im_z1.^2;
    z2_norm_2 = re_z2.^2 + im_z2.^2;
    
%     idx_z1 = (z1_norm_2 > 0);
%     idx_z2 = (z2_norm_2 > 0);
    
    tmp = max(sqrt(z1_norm_2 + z2_norm_2) - tau,0);

    P1 = z1./sqrt(z1_norm_2 + z2_norm_2) .* tmp;
    P2 = z2./sqrt(z1_norm_2 + z2_norm_2) .* tmp;

end

function [P] = Shrinkcomp(D,tau)

% org_sz = size(D);

Dx = real(D);
Dy = imag(D);

% Dx = Dx(:);
% Dy = Dy(:);

Dnorm = sqrt(Dx.^2 + Dy.^2);
% idx = find(Dnorm > 0);
idx_log = (Dnorm > 0);
tmp = max(Dnorm - tau,0);

% size(Dx)

Px = zeros(size(Dx));
Px(idx_log) = (Dx(idx_log)./Dnorm(idx_log)).*tmp(idx_log);
% Px(Dnorm > 0) = (Dx(Dnorm > 0)./Dnorm(Dnorm > 0)).*tmp(Dnorm > 0);
% P(:,:,1) = Px;
%clear Px;
Py = zeros(size(Dy));
Py(idx_log) = (Dy(idx_log)./Dnorm(idx_log)).*tmp(idx_log);
% Py(Dnorm > 0) = (Dy(Dnorm > 0)./Dnorm(Dnorm > 0)).*tmp(Dnorm > 0);
% P(:,:,2) = Py;
%clear Py;

P = complex(Px,Py);

% P = reshape(P, org_sz);

end


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
  if ~isfield(opt,'L1Weight'),
    opt.L1Weight = 1;
  end
  if ~isfield(opt,'GrdWeight'),
    opt.GrdWeight = 1;
  end
  if ~isfield(opt,'Y0'),
    opt.Y0 = [];
  end
  if ~isfield(opt,'U0'),
    opt.U0 = [];
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
    opt.RhoRsdlRatio = 1.2;
  end
  if ~isfield(opt,'RhoScaling'),
    opt.RhoScaling = 100;
  end
  if ~isfield(opt,'AutoRhoScaling'),
    opt.AutoRhoScaling = 1;
  end
  if ~isfield(opt,'RhoRsdlTarget'),
    opt.RhoRsdlTarget = [];
  end
  if ~isfield(opt,'StdResiduals'),
    opt.StdResiduals = 0;
  end
  if ~isfield(opt,'RelaxParam'),
    opt.RelaxParam = 1.8;
  end
  if ~isfield(opt,'NonNegCoef'),
    opt.NonNegCoef = 0;
  end
  if ~isfield(opt,'NoBndryCross'),
    opt.NoBndryCross = 0;
  end
  if ~isfield(opt,'AuxVarObj'),
    opt.AuxVarObj = 0;
  end
  if ~isfield(opt,'HighMemSolve'),
    opt.HighMemSolve = 0;
  end

end