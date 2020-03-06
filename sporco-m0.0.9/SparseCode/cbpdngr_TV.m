function [Y, optinf] = cbpdngr_TV(D, S, lambda, mu, opt)






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
  xsz = [size(S,1) size(S,2) size(D,3) 1];
  hrm = 1;
end

% Compute filters in DFT domain
Df = fft2(D, size(S,1), size(S,2));
grv = [-1 1];
Grf = fft2(grv, size(S,1), size(S,2));
gcv = [-1 1]';
Gcf = fft2(gcv, size(S,1), size(S,2));
if isscalar(opt.GrdWeight),
  opt.GrdWeight = opt.GrdWeight * ones(size(D,3), 1);
end

wgr = reshape(opt.GrdWeight, [1 1 length(opt.GrdWeight)]);
GfW = bsxfun(@times, conj(Grf).*Grf + conj(Gcf).*Gcf, wgr);

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
if opt.HighMemSolve,
  cn = bsxfun(@rdivide, Df, mu*GfW + rho);
  cd = sum(Df.*bsxfun(@rdivide, conj(Df), mu*GfW + rho), 3) + 1.0;
  C = bsxfun(@rdivide, cn, cd);
  clear cn cd;
else
  C = [];
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

% Y0 = zeros([size(S,1), size(S,2), 1, size(S,3)]);
% Y1 = zeros([size(S,1), size(S,2), 1, size(S,3)]);

Y0 = Y2;
Y1 = Y2;

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
% U0 = zeros([size(S,1), size(S,2), 1, size(S,3)]);
% U1 = zeros([size(S,1), size(S,2), 1, size(S,3)]);

U0 = U2;
U1 = U2;


% Main loop
k = 1;
while k <= opt.MaxMainIter && (r > epri | s > edua),
    
    % Solve X subproblem
    Xf = solvedbd_sm(Df, rho*GfW + rho, DSf + rho*(fft2(Y2 - U2) + ...
            bsxfun(@times, conj(Grf), fft2(Y0- U0)) + bsxfun(@times, conj(Gcf), fft2(Y1-U1))), C);
    X = ifft2(Xf);
    
    % See pg. 21 of boyd-2010-distributed
    if opt.RelaxParam == 1,
        Xr = X;
    else
        Xr = opt.RelaxParam*X + (1-opt.RelaxParam)*Y2;
    end
    
    % Solve Y subproblem
    
    Y2 = Shrinkcomp(Xr + U2, (lambda/rho)*opt.L1Weight);
    [Y0, Y1] = block_shrink_cpx(ifft2(Grf.*Xf)+U0, ...
        ifft2(Gcf.*Xf)+U1, mu/rho);
    
    if opt.NonNegCoef,
        Y2(Y2 < 0) = 0;
    end
    if opt.NoBndryCross,
        Y2((end-size(D,1)+2):end,:,:,:) = 0;
        Y2(:,(end-size(D,2)+2):end,:,:) = 0;
    end
    
    % Update dual variable
    U0 = U0 + ifft2(Grf.*Xf) - Y0;
    U1 = U1 + ifft2(Gcf.*Xf) - Y1;
    U2 = U2 + Xr - Y2;
    
    % Compute data fidelity term in Fourier domain (note normalisation)
    if opt.AuxVarObj,
        Yf = fft2(Y2); % This represents unnecessary computational cost
        Jdf = sum(vec(abs(sum(bsxfun(@times,Df,Yf),3)-Sf).^2))/(2*xsz(1)*xsz(2));
        Jl1 = sum(abs(vec(bsxfun(@times, opt.L1Weight, Y2))));
%         Jgr = sum(vec((bsxfun(@times, GfW, conj(Yf).*Yf))))/(2*xsz(1)*xsz(2));
        Jgr = sum(abs(vec(sqrt(conj(Grf.*Yf).*(Grf.*Yf) + ...
            conj(Gcf.*Yf).*(Gcf.*Yf)))))/(2*xsz(1)*xsz(2));
    else
        Jdf = sum(vec(abs(sum(bsxfun(@times,Df,Xf),3)-Sf).^2))/(2*xsz(1)*xsz(2));
        Jl1 = sum(abs(vec(bsxfun(@times, opt.L1Weight, X))));
%         Jgr = sum(vec((bsxfun(@times, GfW, conj(Xf).*Xf))))/(2*xsz(1)*xsz(2));
        Jgr = sum(abs(vec(sqrt(conj(Grf.*Xf).*(Grf.*Xf) + ...
            conj(Gcf.*Xf).*(Gcf.*Xf)))))/(2*xsz(1)*xsz(2));
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


