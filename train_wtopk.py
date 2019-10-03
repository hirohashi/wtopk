import sys, argparse
import numpy as np
import scipy.io
from wtopk import train_wtopk_lacofw

def mat2py_conv(y_src):
    y_dst = y_src.flatten().astype(np.int)
    y_dst -= y_dst.min()
    return y_dst

def get_arguments():
    parser = argparse.ArgumentParser(description='Solving weighted top-k multiclass SVM')
    parser.add_argument('--c_svm', type=float, default=1.0, help='the old regularization parameter for SVM')
    parser.add_argument('--gam_sm', type=float, default=0.0, help='the smoothness parameter of the convex conjufate')
    parser.add_argument('--nepochs', type=float, default=10000, help='the number of local solver iterations')
    parser.add_argument('--method', type=str, default='stdfw', help='stdfw, pfw, afw')
    parser.add_argument('--rho_dist', type=str, default='topk_uniform', choices=['topk_uniform', 'topk_linear','topk_exp',])
    parser.add_argument('--rho_param', type=float, default=1.0, help='a parameter of the rho distribution')
    parser.add_argument('--dataset', type=str, default='./k.demo450_01.mat', help='matlab format dataset')
    parser.add_argument('--thres_gap', type=float, nargs='?', default=1e-3, help='the stopping criteria of duality gap')
    # parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    return args

def main():
    para_root = True
    
    # parse arguments.
    args = get_arguments()
    c_svm = args.c_svm
    rho_dist = args.rho_dist
    rho_param = args.rho_param
    gam_sm = args.gam_sm
    nepochs = args.nepochs
    method = args.method
    thres_gap = args.thres_gap
    fname_dataset = args.dataset
    verbose = True # args.verbose
    
    # load dataset
    mat_in = scipy.io.loadmat(fname_dataset)
    X_tra, y_tra = mat_in['X_tra'], mat_in['y_tra']
    ncts = np.unique(y_tra).size
    
    # set rhos distribution
    if rho_dist == 'topk_uniform':
        k_tk = int(rho_param)
        rhos = np.zeros(ncts)
        rhos[:k_tk] = 1.0
    elif rho_dist == 'topk_linear':
        k_tk = int(rho_param)
        rhos = np.maximum(0, k_tk - np.arange(ncts) )    
    elif rho_dist == 'topk_exp':
        k_tk = int(rho_param)
        rhos = np.zeros(ncts)
        rhos[:k_tk] = np.exp(-1.0*np.arange(k_tk)/k_tk)
    else:
        raise ValueError
    rhos = rhos / rhos.sum()
    
    # set training parameter.
    param = {}
    param['verbose'] = verbose
    param['thres_gap'] = thres_gap
    param['nepochs'] = nepochs
    param['rhos'] = rhos
    param['gam_sm'] = gam_sm
    param['typ_fw'] = method
    param['use_rel_gap'] = True
    
    y_tra = mat2py_conv(y_tra)
    ntras = y_tra.size
    lam = 1. / (ntras * c_svm);
    W_est, tra_log = train_wtopk_lacofw(X_tra, y_tra, lam, param)
    
    print('\n' * 3);
    print('---')
    print('The weight matrix is obtained as follows:');
    print(W_est)
    
if __name__ == '__main__':
    main()

