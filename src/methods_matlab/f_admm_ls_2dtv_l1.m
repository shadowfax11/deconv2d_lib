function [s_est, optim_info] = f_admm_ls_2dtv_l1(b, h, optim, s_init, disp_info)
%F_ADMM_2DTV_L1 Solve: min_s 0.5*||b - conv2(h,s)||_2^2 +
%lambda_L1*||s||_1^1 + lambda_TV*||TV(s)||_1^1
%   b: Blurred image (2D array)
%   h: PSF (2D array)
%   optim: (cell) contains optimization-specific information
%   x_init: (2D array) initialization
%   disp_info: (cell) contains display-related information

[py, px] = size(h);
[Ny, Nx] = size(b);

b = padarray(b, [(py)/2, (px)/2], 0, 'both');
h = padarray(h, [(Ny)/2, (Nx)/2], 0, 'both');

if nargin<3
    optim = {}; 
    optim.max_iters = 25;
    optim.lambda_L1 = 0;
    optim.lambda_TV = 0;
    s_init = zeros(size(b));
    disp_info = {};
    disp_info.disp_flag = 1;
    disp_info.disp_freq = 5;
end
if nargin<4
    s_init = zeros(size(b));
    disp_info = {};
    disp_info.disp_flag = 1;
    disp_info.disp_freq = 5;
end
if nargin<5
    disp_info = {};
    disp_info.disp_flag = 1;
    disp_info.disp_freq = 5;
end
if isempty(s_init)
    s_init = zeros(size(b));
end

if ~isfield(optim,'max_iters')
    optim.max_iters = 25;
end
if ~isfield(optim,'lambda_L1')
    optim.lambda_L1 = 0;
end
if ~isfield(optim,'lambda_TV')
    optim.lambda_TV = 0;
end
if ~isfield(optim,'resid_tol')
    optim.resid_tol = 1.5;
end
if ~isfield(optim,'tau_inc')
    optim.tau_inc = 1.1;
end
if ~isfield(optim,'tau_dec')
    optim.tau_dec = 1.1;
end
if ~isfield(optim,'decay_factor')
    optim.decay_factor = 1;
end
fprintf("Performing ADMM-based deconvolution\n");
disp(optim);

% aux functions
pad2d = @(x) padarray(x, [py/2,px/2], 0, 'both');
crop2d = @(x) x(py/2+1:end-py/2, px/2+1:end-px/2);
vec = @(x) reshape(x, numel(x), 1);

% filter operators
fftshift2 = @(x) fftshift(fftshift(x,1),2);
ifftshift2 = @(x) ifftshift(ifftshift(x,1),2);

Fx2 = @(x) fft2(fftshift2(x));
FiltX2 = @(H,x) real(ifftshift2(ifft2(H.*Fx2(x))));

% aux functions for FFT-based convolution
H = Fx2(h);
H_conj = conj(H);

Hfor = @(x) FiltX2(H, x);       % forward operator
Hadj = @(x) FiltX2(H_conj, x);  % adjoint operator

% TV forward and adjoint operators
% x-grad filter
kx = zeros(size(h));
kx(1,1,:) = 1;
kx(1,2,:) = -1;
Kx = fft2(kx);
% y-grad filter
ky = zeros(size(h));
ky(1,1,:) = 1;
ky(2,1,:) = -1;
Ky = fft2(ky);

Psi = @(x) deal(FiltX2(Kx,x), FiltX2(Ky,x));
PsiT = @(P1,P2) FiltX2(conj(Kx),P1) + FiltX2(conj(Ky),P2);
% set up PsiTPsi
L = zeros(Ny+py, Nx+px);
L(1,1) = 4;
L(1,2) = -1;
L(2,1) = -1;
L(1,end) = -1;
L(end,1) = -1;

% cost trackers
f_fid = []; 
f_rtv = []; 
f_rl1 = [];
f_obj = [];
f_alt = [];

% residue trackers
prim_res_u = [];
dual_res_u = [];
prim_res_v = [];
dual_res_v = [];
prim_res_w = [];
dual_res_w = [];
prim_res_q = [];
dual_res_q = [];

% set up optimization variables
MAX_ITERS = optim.max_iters; 
lambda_L1 = optim.lambda_L1;
lambda_TV = optim.lambda_TV;
resid_tol = optim.resid_tol; 
tau_inc = optim.tau_inc; 
tau_dec = optim.tau_dec;

if disp_info.disp_flag
    fh1 = figure; 
    fh2 = figure; 
end

% set up ADMM iterations
HtH = abs(H.*H_conj);
PsiTPsi = real(fft2(L));
CtC = pad2d(ones(Ny,Nx,'like',b));

rho_u = 1;
rho_v = 1;
rho_w = 1;
rho_q = 1;

u_mult = 1./(CtC + rho_u);
F_mult = 1./(rho_u*HtH + rho_v*PsiTPsi + rho_w + rho_q);

% initialize
s_old = s_init;
Hs_old = Hfor(s_old);
[Dy_s_old, Dx_s_old] = Psi(s_old);

dual_u_old = s_old;
dual_vy_old = s_old;
dual_vx_old = s_old;
dual_w_old = s_old;
dual_q_old = s_old;

iter = 0;
tic; 
fprintf("Starting ADMM optimization\n");
while iter < MAX_ITERS
    iter = iter+1;
    
    % u-update
    u_new = u_mult.*(b + rho_u*Hs_old + dual_u_old);

    % v-update
    vy_new = shrinkageOp(Dy_s_old + dual_vy_old/rho_v, lambda_TV/rho_v);
    vx_new = shrinkageOp(Dx_s_old + dual_vx_old/rho_v, lambda_TV/rho_v);

    % w-update
    w_new = max(s_old + dual_w_old/rho_w, 0);

    % q-update
    q_new = shrinkageOp(s_old + dual_q_old/rho_q, lambda_L1/rho_q);

    % s-update
    r_new = rho_u*Hadj(u_new - dual_u_old/rho_u) + ...
            rho_v*PsiT(vy_new - dual_vy_old/rho_v, vx_new - dual_vx_old/rho_v) + ...
            (rho_w*w_new - dual_w_old) + ...
            (rho_q*q_new - dual_q_old);

    s_new = FiltX2(F_mult, r_new);
    
    % dual_u update
    Hs_new = Hfor(s_new); 
    rpU = Hs_new - u_new;
    dual_u_new = dual_u_old + rho_u*rpU;
    
    % dual_z update
    [Dy_s_new, Dx_s_new] = Psi(s_new); 
    rpVy = Dy_s_new - vy_new; 
    rpVx = Dx_s_new - vx_new; 
    dual_vy_new = dual_vy_old + rho_v*rpVy;
    dual_vx_new = dual_vx_old + rho_v*rpVx;
    
    % dual_w update
    rpW = s_new - w_new;
    dual_w_new = dual_w_old + rho_w*rpW;
    
    % dual_q update
    rpQ = s_new - q_new;
    dual_q_new = dual_q_old + rho_q*rpQ;
    
    % costs update
    f_fid(iter) = 0.5*norm(b - Hs_new, 'fro')^2;
    f_rtv(iter) = lambda_TV*sum( sum(abs(Dy_s_new(:))) + sum(abs(Dx_s_new(:))) );
    f_rl1(iter) = lambda_L1*norm(s_new(:),1);
    f_obj(iter) = f_fid(iter) + f_rtv(iter) + f_rl1(iter);
    f_alt(iter) = 0.5*(norm(crop2d(b)-crop2d(u_new),'fro')^2) ...
                    + lambda_TV*( sum(abs(vy_new(:))) + sum(abs(vx_new(:))) ) ...
                    + lambda_L1*sum(abs(q_new(:)));
    
    % primal residuals update
    prim_res_u(iter) = norm(rpU(:));
    prim_res_v(iter) = sqrt(norm(rpVy(:))^2 + norm(rpVx(:))^2);
    prim_res_w(iter) = norm(rpW(:));
    prim_res_q(iter) = norm(rpQ(:));

    % dual residuals updates
    dual_res_u(iter) = rho_u*norm(Hs_new(:) - Hs_old(:));
    dual_res_v(iter) = rho_v*sqrt( norm(Dy_s_new(:)-Dy_s_old(:))^2 + ...
                    norm(Dx_s_new(:)-Dx_s_old(:))^2 );
    dual_res_w(iter) = rho_w*norm(s_new(:) - s_old(:));
    dual_res_q(iter) = rho_q*norm(s_new(:) - s_old(:));
    
    % rho updates
    [rho_u, rho_u_update] = muUpdater(rho_u,prim_res_u(iter),dual_res_u(iter),resid_tol,tau_inc,tau_dec);
    [rho_v, rho_v_update] = muUpdater(rho_v,prim_res_v(iter),dual_res_v(iter),resid_tol,tau_inc,tau_dec);
    [rho_w, rho_w_update] = muUpdater(rho_w,prim_res_w(iter),dual_res_w(iter),resid_tol,tau_inc,tau_dec);
    [rho_q, rho_q_update] = muUpdater(rho_q,prim_res_q(iter),dual_res_q(iter),resid_tol,tau_inc,tau_dec);
    if rho_u_update || rho_v_update || rho_w_update || rho_q_update
        u_mult = 1./(CtC + rho_u);
        F_mult = 1./(rho_u*HtH + rho_v*PsiTPsi + rho_w + rho_q);
        if disp_info.disp_flag
            disp([rho_u, rho_v, rho_w, rho_q]);
        end
    else
        u_mult = 1./(CtC + rho_u);
    end
    
    % update old estimates with new
    Hs_old = Hs_new;
    Dy_s_old = Dy_s_new;
    Dx_s_old = Dx_s_new;
    
    s_old = s_new;
    
    dual_u_old = dual_u_new;
    dual_vy_old = dual_vy_new;
    dual_vx_old = dual_vx_new; 
    dual_w_old = dual_w_new;
    dual_q_old = dual_q_new;
    
    lambda_TV = lambda_TV/optim.decay_factor;
    lambda_L1 = lambda_L1/optim.decay_factor;
    
    if disp_info.disp_flag
        if ~mod(iter, disp_info.disp_freq) || (iter==MAX_ITERS)
            figure(fh1);
            subplot(1,3,1); imagesc(b); title('Image'); colorbar; axis image; 
            subplot(1,3,2); imagesc(s_new); title('Scene Reconstruction'); colorbar; axis image; 
            subplot(1,3,3); imagesc(abs(b - Hs_old)); title('Error'); colorbar; axis image;
            drawnow;
            figure(fh2);
            subplot(2,2,1); semilogy(1:iter, [prim_res_u',prim_res_v',prim_res_w',prim_res_q']); title("primal residues");
            legend('FID','TV','NN','L1');
            subplot(2,2,2); semilogy(1:iter, [dual_res_u',dual_res_v',dual_res_w',dual_res_q']); title("dual residues");
            legend('FID','TV','NN','L1');
            subplot(2,2,3); semilogy(1:iter, [f_fid', f_rtv', f_rl1', f_obj', f_alt']); title("Costs");
            legend('FID','TV','L1','OBJ','ALT');
            drawnow;
        end
    end
    if ~mod(iter, 5) || (iter==MAX_ITERS) 
        fprintf("%d out of %d iterations done\n", iter, MAX_ITERS);
    end
end
toc;

s_est = s_new;

optim_info = {};
optim_info.pad2d = pad2d;
optim_info.crop2d = crop2d;
optim_info.optim = optim;
optim_info.prim_res_u = prim_res_u;
optim_info.prim_res_v = prim_res_v;
optim_info.prim_res_q = prim_res_q;
optim_info.prim_res_w = prim_res_w;
optim_info.dual_res_u = dual_res_u;
optim_info.dual_res_v = dual_res_v;
optim_info.dual_res_q = dual_res_q;
optim_info.dual_res_w = dual_res_w;
end

