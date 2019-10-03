import sys, argparse
import numpy as np
import scipy.io
import lacofw

fdot = lambda A,B : (A*B).sum()
fnorm = lambda V : fdot(V,V)

def reshape(x, *shape):
    return x.reshape(*shape)

def get_wtopk_loss_grad(scomat,y_tra,rhos,require_grad=True):
    # Compute the weighted top-k loss.
    ncts, ntras = scomat.shape
    monotonic_idx = np.arange(ntras)
    scodmat =  scomat - scomat[y_tra,monotonic_idx]
    scodmat = scodmat + 1
    scodmat[y_tra, monotonic_idx] -= 1
    rmat_arg = np.argsort(-scodmat, axis=0)
    rmat_srt = np.take_along_axis(scodmat, rmat_arg, axis=0)
    loss = np.maximum(0, rhos @ rmat_srt)
    if not require_grad:
        return loss
    
    # Compute the subgradient for the loss.
    r_suf = np.where(loss>0)[0]
    grad1mat = np.zeros((ncts,ntras))
    for i_tra in r_suf:
        y1 = y_tra[i_tra]
        r1 = rmat_arg[:,i_tra]; 
        grad1d = grad1mat[:,i_tra]
        grad1d[r1] = rhos
        grad1d[y1] -= grad1d.sum()
        grad1mat[:,i_tra] = grad1d
    return loss, grad1mat

def wtopk_objgap(x_new, fworac, X_tra, y_tra, lam, param):
    nfeas, ntras = X_tra.shape
    lamn = lam * ntras; ilamn = 1./lamn;
    ncts, rhos = [ param[key] for key in ['ncts', 'rhos'] ]
    
    # borrow variables from the FW-Oracle for fast computation
    obj_d, reg_d, loss_p = [ fworac[key] for key in ['obj', 'reg_d', 'primal_loss'] ]
    
    # compute dual objective (max(D) is treated as min(-D) in the FW-algo)
    obj_d = -obj_d
    
    # compute primal objective
    reg_p = -reg_d
    loss_p = loss_p.mean()
    obj_p  = reg_p + loss_p
    
    objgap = obj_p - obj_d
    
    fworac['obj_primal'] = obj_p
    fworac['obj_dual'] = obj_d
    fworac['obj_gap'] = objgap
    
    return objgap

def wtopk_fworac(x_fw, X_tra, y_tra, lam, params_tk):
    ''' Obtain frank-wolfe oracke for wtopk loss and regularizer '''
    nfeas, ntras = X_tra.shape
    lamn = lam * ntras
    ilamn = 1./lamn
    
    ncts, rhos, gam_sm = [ params_tk[k] for k in ['ncts', 'rhos', 'gam_sm']]
    
    A_old = reshape(x_fw, ncts,ntras) 
    mu_sm2  = gam_sm * lamn * ntras
    W_old = (X_tra @ A_old.T) * ilamn
    
    scomat = W_old.T @ X_tra + mu_sm2 * ilamn * A_old
    E_ct = np.eye(ncts)[y_tra].T
    primal_loss, primal_grad  = get_wtopk_loss_grad(scomat,y_tra,rhos)
    umat = -primal_grad
    grad1 = (E_ct-scomat) / ntras
    pnV   = fdot(scomat,A_old) * ilamn
    reg_d    = -0.5*lam*pnV
    loss_d   = fdot(E_ct,A_old) / ntras
    obj_d = reg_d + loss_d
    fworac = {}
    fworac['primal_loss'] = primal_loss;
    fworac['primal_grad'] = primal_grad;
    fworac['reg_d'] = reg_d;
    fworac['obj'] = -obj_d;
    fworac['grad'] = -reshape(grad1,-1)   # Gradient for FW. 
    fworac['u_old'] = reshape(umat, -1)   # Solution of LMO. 
    fworac['fh_lnsrch'] = lambda dvec, gam_mx : wtopk_do_lnsrch(dvec, gam_mx, lamn, E_ct, scomat, X_tra,mu_sm2)
    fworac['W_old'] = W_old
    fworac['A_old'] = A_old
    fworac['scomat'] = scomat
    return fworac

def wtopk_do_lnsrch(dvec,gam_max,lamn,E_ct,scomat,X_tra,mu_sm2):
    ''' Line search for the FW algorithm '''
    qmat = reshape( dvec, *E_ct.shape )
    nume1 = lamn * fdot( qmat, E_ct - scomat)
    deno1 = fnorm(qmat @ X_tra.T) + fnorm(qmat) * mu_sm2
    gam_opt = np.clip(nume1/deno1, 0, gam_max)
    return gam_opt

def train_wtopk_lacofw(X_tra, y_tra, lam, params):
    ''' Train the weighted top-k svm by the lacoste-julien's frank-wolfe method.'''
    nfeas, ntras = X_tra.shape
    lamn = lam * ntras;
    ilamn = 1./lamn;
    
    ncts = params['rhos'].size
    ntras = y_tra.size;
    A_init = np.zeros((ncts,ntras));
    
    # Set parameters for weighted top-k SVM.
    params_wtk = {}
    params_wtk['ncts'] = ncts
    params_wtk['rhos'] = params['rhos']
    params_wtk['gam_sm'] = params['gam_sm']
    fh_fworac = lambda x : wtopk_fworac( x, X_tra, y_tra, lam, params_wtk)
    fh_objgap = lambda x, fworac : wtopk_objgap( x, fworac, X_tra, y_tra, lam, params_wtk)
    
    # Initialization
    param_lacofw = {}
    param_lacofw['verbose'] = params['verbose']
    param_lacofw['rec_mode'] = params.get('rec_mode', 'all')
    param_lacofw['typ_fw'] = params['typ_fw']
    param_lacofw['niters'] = params['nepochs'] 
    param_lacofw['fh_fworac'] = fh_fworac
    param_lacofw['fh_objgap'] = fh_objgap
    param_lacofw['thres_gap'] = params['thres_gap']
    param_lacofw['use_rel_gap'] = params['use_rel_gap']
    param_lacofw['x_init'] = reshape(A_init, -1)
    res_fw = lacofw.do_lacofw(param_lacofw)
    x_est = res_fw['x_est']
    fworac = fh_fworac(x_est)
    # fh_objgap(x_est, fworac)
    
    res = {}
    res['gs'] = -res_fw['fs'];
    res['fs'] = res_fw['gaps'] + res['gs'];
    res['gaps'] = res_fw['gaps']
    res['tstamp'] = res_fw['tms'];
    res['teas_rec'] = res_fw['iters_rec'];
    res['iter'] = res_fw['iter'];
    if 'vertex' in res_fw:
        res['vertex'] = res_fw['vertex']
    res['W_est'] = fworac['W_old']
    res['A_est'] = fworac['A_old']
    
    return res['W_est'], res
