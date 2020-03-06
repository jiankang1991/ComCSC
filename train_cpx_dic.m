%%%% train dictionary based on sporco

% dbstop if error

clear all
close all

load train_cpxs
% load train_cpxs_mountains

rng('default');
rng(2);

batch_size = 80;


cpx = train_cpx(:,:,1:batch_size);


% cpx_norm = cpx - min(min(cpx,[],1),[],2) ./(max(max(cpx,[],1),[],2));


% figure;
% imagesc(angle(cpx(:,:,1)));
% colormap jet
% 
% figure;
% imagesc(abs(cpx(:,:,1)));
% colormap jet


%% 

num_kernel = 96;
kernel_sz = 20;

% Construct initial dictionary
D0 = zeros(kernel_sz,kernel_sz,num_kernel, 'single');
% D0(3:6,3:6,:) = single(randn(4,4,num_kernel) + 1i*randn(4,4,num_kernel));

D0 = single(randn(kernel_sz,kernel_sz,num_kernel) + 1i*randn(kernel_sz,kernel_sz,num_kernel));


% Set up cbpdndl parameters
lambda = 0.2;
opt = [];
opt.Verbose = 1;
opt.MaxMainIter = 200;
opt.rho = 50*lambda + 0.5;
opt.sigma = size(cpx,3);
opt.AutoRho = 0;
opt.AutoRhoPeriod = 10;
opt.AutoSigma = 0;
opt.AutoSigmaPeriod = 10;
% opt.XRelaxParam = 1.8;
% opt.DRelaxParam = 1.8;

opt.XRelaxParam = 1;
opt.DRelaxParam = 1;

opt.display = 1;

% Do dictionary learning without mask
% [D, X, optinf] = cbpdndl(D0, cpx, lambda, opt);

% with mask

pad_num = size(D0,1) - 1;

padded_noisy_cpx = padarray(cpx, [pad_num pad_num], 'both');
W = ones(size(cpx));
W = padarray(W, [pad_num pad_num], 'both');

tic;
[D, X, optinf] = cbpdndlms(D0, cpx, lambda, opt);
CCDL_t = toc;

%

D_img = tiledict(D);

figure;
imagesc(real(D_img));colormap gray;
% 
% 
% D0_img = tiledict(D0);
% 
% figure;
% imagesc(real(D0_img));colormap grey;

% figure
% plot(optinf.loss, 'x-');

% save(sprintf('filters_%d_num_train_sample%d_filtersz%d_masked.mat', size(D0,3), batch_size, size(D0,1)), 'D', 'optinf');










