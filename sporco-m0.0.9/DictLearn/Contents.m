% Dictionary learning functions
%
% cmod          -- Constrained Method of Optimal Directions (MOD)
% bpdndl        -- BPDN Dictionary Learning
%
% Convolutional dictionary learning functions (batch learning)
%
% ccmod         -- Convolutional Constrained MOD
% cbpdndl       -- Convolutional BPDN Dictionary Learning
% cbpdndlcns    -- Convolutional BPDN Dictionary Learning (ADMM Consensus)
% cbpdndlmd     -- Convolutional BPDN Dictionary Learning with Mask Decoupling
% cbpdndlms     -- Convolutional BPDN Dictionary Learning with Mask Simulation
%
% Convolutional dictionary learning functions (online learning)
%
% olcdl_surfnc_freq  -- Online CDL via the surrogate function method
%                       (frequency domain computation)
% olcdl_surfnc_sprs  -- Online CDL via the surrogate function method
%                       (sparse matrix computation)
% olcdl_surfnc_sprs_msk -- Online CDL via the surrogate function method, with
%                       data fidelity term mask (sparse matrix computation)
% olcdl_sgd_freq     -- Online CDL via stochastic gradient descent
%                       (frequency domain computation)
% olcdl_sgd_sprs     -- Online CDL via stochastic gradient descent
%                       (sparse matrix computation)
% olcdl_sgd_freq_msk -- Online CDL via stochastic gradient descent, with
%                       data fidelity term mask (frequency domain computation)
% olcdl_sgd_sprs_msk -- Online CDL via stochastic gradient descent, with
%                       data fidelity term mask (sparse matrix computation)
%
% Support functions
%
% solvemdbi_cg      -- Solve a multiple diagonal block linear system via CG
% solvemdbi_ism     -- Solve a multiple diagonal block linear system via
%                      iterated Sherman-Morrison
