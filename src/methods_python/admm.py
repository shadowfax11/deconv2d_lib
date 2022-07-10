import os
import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt

def shrinkageOp(x, lmbd):
    x[np.where(np.abs(x)<=lmbd)] = 0
    x = x - lmbd*((x>lmbd).astype('float'))
    x = x + lmbd*((x<-1*lmbd).astype('float'))
    return x

def muUpdater(mu, primal_res, dual_res, resid_tol, tau_inc, tau_dec):
    if primal_res > resid_tol*dual_res:
        mu_out = mu*tau_inc
        mu_update = 1
    elif dual_res > resid_tol*primal_res:
        mu_out = mu/tau_dec
        mu_update = -1
    else:
        mu_out = mu
        mu_update = 0
    return mu_out, mu_update

def admm_ls_2dtv_l1(b, h, optim=None, s_init=None, disp_info=None):
    """
    Solves the following optimization problem using ADMM
    min_s 0.5*||b - conv2(h,s)||_2^2 + lambda_L1*||s||_1^1 + lambda_TV*||TV(s)||_1^1
    b(numpy.ndarray): Blurred image (2D array)
    h(numpy.ndarray): PSF (2D array)
    optim(dict): contains optimization-specific information
    s_init(numpy.ndarray): Initialization. Default is zero signal.
    disp_info(dict): contains display-specific information.
    """
    py, px = h.shape
    Ny, Nx = b.shape

    b = np.pad(b,((int(py/2),int(py/2)),(int(px/2),int(px/2))),'constant')
    h = np.pad(h,((int(Ny/2),int(Ny/2)),(int(Nx/2),int(Nx/2))),'constant')

    if optim is None:
        optim = {}
        optim['max_iters'] = 25
        optim['lambda_L1'] = 0
        optim['lambda_TV'] = 0
    if s_init is None:
        s_init = np.zeros_like(b)
    if disp_info is None:
        disp_info = {}
        disp_info['disp_flag'] = 1
        disp_info['disp_freq'] = 5
    
    if not "max_iters" in optim:
        optim['max_iters'] = 25
    if not "lambda_L1" in optim:
        optim['lambda_L1'] = 0
    if not "lambda_TV" in optim:
        optim['lambda_TV'] = 0
    if not "resid_tol" in optim:
        optim['resid_tol'] = 1.5
    if not "tau_inc" in optim:
        optim['tau_inc'] = 1.1
    if not "tau_dec" in optim:
        optim['tau_dec'] = 1.1
    if not "decay_factor" in optim:
        optim['decay_factor'] = 1
    print("Performing ADMM-based deconvolution: LS + L1 + TV")
    print(optim)

    def pad2d(x):
        return np.pad(x, ((int(py/2),int(py/2)),(int(px/2),int(px/2))), 'constant')
    def crop2d(x):
        return x[int(py/2):-int(py/2), int(px/2):-int(px/2)]
    def Fx2(x):
        return fft.fft2(fft.fftshift(x))
    def FiltX2(H,x):
        return np.real(fft.ifftshift(fft.ifft2(H*Fx2(x))))
    
    # forward and adjoint operators
    H = Fx2(h)
    H_conj = np.conjugate(H)
    def Hfor(x):
        return FiltX2(H, x)
    def Hadj(x):
        return FiltX2(H_conj, x)
    
    # TV forward and adjoint operators
    kx = np.zeros_like(h)
    kx[0,0] = 1
    kx[0,1] = -1
    Kx = fft.fft2(kx)
    ky = np.zeros_like(h)
    ky[0,0] = 1
    ky[1,0] = -1
    Ky = fft.fft2(ky)

    def Psi(x):
        return FiltX2(Kx,x), FiltX2(Ky,x)
    def PsiT(p1,p2):
        return FiltX2(np.conjugate(Kx),p1) + FiltX2(np.conjugate(Ky),p2) 
    L = np.zeros((Ny+py, Nx+px))
    L[0,0] = 4
    L[0,1] = -1
    L[1,0] = -1
    L[0,-1] = -1
    L[-1,0] = -1

    # set up optimization variables
    MAX_ITERS = optim['max_iters']
    lambda_L1 = optim['lambda_L1']
    lambda_TV = optim['lambda_TV']
    resid_tol = optim['resid_tol']
    tau_inc = optim['tau_inc']
    tau_dec = optim['tau_dec']

    # cost trackers
    f_fid = np.zeros((MAX_ITERS,))
    f_rtv = np.zeros((MAX_ITERS,))
    f_rl1 = np.zeros((MAX_ITERS,))
    f_obj = np.zeros((MAX_ITERS,))
    f_alt = np.zeros((MAX_ITERS,))

    # residue trackers
    prim_res_u = np.zeros((MAX_ITERS,))
    dual_res_u = np.zeros((MAX_ITERS,))
    prim_res_v = np.zeros((MAX_ITERS,))
    dual_res_v = np.zeros((MAX_ITERS,))
    prim_res_w = np.zeros((MAX_ITERS,))
    dual_res_w = np.zeros((MAX_ITERS,))
    prim_res_q = np.zeros((MAX_ITERS,))
    dual_res_q = np.zeros((MAX_ITERS,))

    if disp_info['disp_flag']:
        plt.ion()
        fh1 = plt.figure()
        ax1 = fh1.add_subplot(131)
        im1 = ax1.imshow((b))
        ax1.set_title('Image')
        ax2 = fh1.add_subplot(132)
        im2 = ax2.imshow(np.zeros_like(b))
        ax2.set_title('Scene Reconstruction')
        ax3 = fh1.add_subplot(133)
        im3 = ax3.imshow(np.abs(b))
        ax3.set_title('Error')


    # set up ADMM iterations
    HtH = np.abs(H*H_conj)
    PsiTPsi = np.real(fft.fft2(L))
    CtC = pad2d(np.ones((Ny,Nx)))

    rho_u = 1
    rho_v = 1
    rho_w = 1
    rho_q = 1

    u_mult = 1./(CtC + rho_u)
    F_mult = 1./(rho_u*HtH + rho_v*PsiTPsi + rho_w + rho_q)

    # initialize
    s_old = s_init
    Hs_old = Hfor(s_old)
    Dy_s_old, Dx_s_old = Psi(s_old)

    dual_u_old = s_old
    dual_vy_old = s_old
    dual_vx_old = s_old
    dual_w_old = s_old
    dual_q_old = s_old

    iter = 0
    print("Starting ADMM optimization")
    while (iter<MAX_ITERS):
        iter = iter + 1

        # u-update
        u_new = u_mult*(b + rho_u*Hs_old + dual_u_old)

        # v-update
        vy_new = shrinkageOp(Dy_s_old + dual_vy_old/rho_v, lambda_TV/rho_v)
        vx_new = shrinkageOp(Dx_s_old + dual_vx_old/rho_v, lambda_TV/rho_v)

        # w-update
        w_new = np.maximum(s_old + dual_w_old/rho_w, 0)

        # q-update
        q_new = shrinkageOp(s_old + dual_q_old/rho_q, lambda_L1/rho_q)

        # s-update
        r_new = rho_u*Hadj(u_new - dual_u_old/rho_u) + \
            rho_v*PsiT(vy_new - dual_vy_old/rho_v, vx_new - dual_vx_old/rho_v) + \
            (rho_w*w_new - dual_w_old) + \
            (rho_q*q_new - dual_q_old)
        s_new = FiltX2(F_mult, r_new)

        # dual_u update
        Hs_new = Hfor(s_new)
        rpU = Hs_new - u_new
        dual_u_new = dual_u_old + rho_u*rpU

        # dual_v update
        Dy_s_new, Dx_s_new = Psi(s_new)
        rpVy = Dy_s_new - vy_new
        rpVx = Dx_s_new - vx_new
        dual_vy_new = dual_vy_old + rho_v*rpVy
        dual_vx_new = dual_vx_old + rho_v*rpVx

        # dual_w update
        rpW = s_new - w_new
        dual_w_new = dual_w_old + rho_w*rpW

        # dual_q update
        rpQ = s_new - q_new
        dual_q_new = dual_q_old + rho_q*rpQ

        # costs update
        f_fid[iter-1] = 0.5*np.linalg.norm(b-Hs_new, ord='fro')**2
        f_rtv[iter-1] = lambda_TV*np.sum(np.sum(np.abs(Dy_s_new)) + np.sum(np.abs(Dx_s_new)))
        f_rl1[iter-1] = lambda_L1*np.sum(np.abs(s_new))
        f_obj[iter-1] = f_fid[iter-1] + f_rtv[iter-1] + f_rl1[iter-1]
        f_alt[iter-1] = 0.5*np.linalg.norm(crop2d(b)-crop2d(u_new),ord='fro')**2 + \
                        lambda_TV*np.sum(np.sum(np.abs(vy_new)) + np.sum(np.abs(vx_new))) + \
                        lambda_L1*np.sum(np.abs(q_new))

        # primal residuals update
        prim_res_u[iter-1] = np.sqrt(np.sum(rpU**2))
        prim_res_v[iter-1] = np.sqrt(np.sum(rpVy**2)+np.sum(rpVx**2))
        prim_res_w[iter-1] = np.sqrt(np.sum(rpW**2))
        prim_res_q[iter-1] = np.sqrt(np.sum(rpQ**2))

        # dual residuals update
        dual_res_u[iter-1] = rho_u*np.sqrt(np.sum((Hs_new-Hs_old)**2))
        dual_res_v[iter-1] = rho_v*np.sqrt(np.sum((Dy_s_new-Dy_s_old)**2)+np.sum((Dx_s_new-Dx_s_old)**2))
        dual_res_w[iter-1] = rho_w*np.sqrt(np.sum((s_new-s_old)**2))
        dual_res_q[iter-1] = rho_q*np.sqrt(np.sum((s_new-s_old)**2))

        # rho update
        rho_u, rho_u_update = muUpdater(rho_u,prim_res_u[iter-1],dual_res_u[iter-1],resid_tol,tau_inc,tau_dec)
        rho_v, rho_v_update = muUpdater(rho_v,prim_res_v[iter-1],dual_res_v[iter-1],resid_tol,tau_inc,tau_dec)
        rho_w, rho_w_update = muUpdater(rho_w,prim_res_w[iter-1],dual_res_w[iter-1],resid_tol,tau_inc,tau_dec)
        rho_q, rho_q_update = muUpdater(rho_q,prim_res_q[iter-1],dual_res_q[iter-1],resid_tol,tau_inc,tau_dec)
        if rho_u_update or rho_v_update or rho_w_update or rho_q_update:
            u_mult = 1./(CtC + rho_u)
            F_mult = 1./(rho_u*HtH + rho_v*PsiTPsi + rho_w + rho_q)
            print("{0:.3f} {1:.3f} {2:.3f} {3:.3f}".format(rho_u, rho_v, rho_w, rho_q))
        else:
            u_mult = 1./(CtC + rho_u)

        # update old estimates with new estimates
        Hs_old = Hs_new
        Dy_s_old = Dy_s_new
        Dx_s_old = Dx_s_new

        s_old = s_new

        dual_u_old = dual_u_new
        dual_vy_old = dual_vy_new
        dual_vx_old = dual_vx_new
        dual_w_old = dual_w_new
        dual_q_old = dual_q_new

        lambda_TV = lambda_TV/optim['decay_factor']
        lambda_L1 = lambda_L1/optim['decay_factor']

        if disp_info['disp_flag']:
            if (iter%disp_info['disp_freq']==0) or (iter==MAX_ITERS):
                im1.set_data(b)
                im2.set_data(s_new)
                im3.set_data(np.abs(b-Hs_old))
                fh1.canvas.draw()
                fh1.canvas.flush_events()
        
        if (iter%5==0) or (iter==MAX_ITERS):
            print("{:3d} out of {:3d} iterations done".format(iter, MAX_ITERS))
    
    plt.ioff()
    iters = np.arange(1,MAX_ITERS+1,1)
    if disp_info['disp_flag']:
        fh2 = plt.figure()
        ax21 = fh2.add_subplot(221)
        ax21.semilogy(iters, prim_res_u, 'b-', iters, prim_res_v, 'r-', iters, prim_res_w, 'g-', iters, prim_res_q, 'k-')
        ax21.set_title('Primal residues')
        ax22 = fh2.add_subplot(222)
        ax22.semilogy(iters, dual_res_u, 'b-', iters, dual_res_v, 'r-', iters, dual_res_w, 'g-', iters, dual_res_q, 'k-')
        ax22.set_title('Dual residues')
        ax23 = fh2.add_subplot(223)
        ax23.semilogy(iters, f_fid, 'b-', iters, f_rtv, 'r-', iters, f_rl1, 'g-', iters, f_obj, 'm-', iters, f_alt, 'k-')

    s_est = s_new

    optim_info = {}
    optim_info['optim'] = optim
    optim_info['prim_res_u'] = prim_res_u
    optim_info['prim_res_v'] = prim_res_v
    optim_info['prim_res_w'] = prim_res_w
    optim_info['prim_res_q'] = prim_res_q
    optim_info['dual_res_u'] = dual_res_u
    optim_info['dual_res_v'] = dual_res_v
    optim_info['dual_res_w'] = dual_res_w
    optim_info['dual_res_q'] = dual_res_q

    return s_est, optim_info, pad2d, crop2d