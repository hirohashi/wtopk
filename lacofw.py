import time, sys
import numpy as np

def do_lacofw(param):    
    fh_fworac = param['fh_fworac']
    fh_objgap = param.get('fh_objgap', None)
    
    x_init = param['x_init']
    assert( x_init.ndim == 1 )
    nfeas = x_init.size
    
    typ_fw = param.get('typ_fw', 'pfw')
    thres_gap1 = param.get('thres_gap1', 1e-15)
    thres_gap2 = param['thres_gap']
    use_rel_gap = param['use_rel_gap']
    niters = int( param.get('niters', 1e5) )
    verbose = param.get('verbose', False)
    rec_mode = param.get('rec_mode', 'log')
    
    if rec_mode == 'all':
        iters_rec = np.arange(1,niters+1)
    elif rec_mode == 'log':
        iters_rec = np.ceil( np.logspace(0, np.log10(niters), 100) ).astype(np.int)
        iters_rec = np.unique(iters_rec)
        iters_rec = iters_rec[ iters_rec < niters ]
    else:
        raise ValueError

    typs_fw = ['stdfw','pfw','afw']
    is_stdfw, is_pfw, is_afw = [ typ_fw == x for x in typs_fw ]
    assert( typ_fw in typs_fw )
    
    x_new = x_init
    n_rec = iters_rec.size
    tms, fs, gaps = [ np.zeros(n_rec) for _ in range(3) ]
    
    if not is_stdfw:
        vertex = np.zeros(n_rec)
        LacoList = list()
        LacoList.append( [1.0, x_new] )
    
    time_list = list()
    iter_cnt = 0
    rec_idx = 0
    finished = False
    tm_start = time.time()
    tm_ignore = 0.0
    while finished == False:
        iter_cnt += 1
        tm_current = time.time() - tm_start
        
        x_old = x_new
        fworac1   = fh_fworac(x_old); 
        obj_p, grad1, u_old, fh_lnsrch = [ fworac1[key] for key in ['obj', 'grad', 'u_old', 'fh_lnsrch']]; 
        
        tm_ignore0 = time.time()
        if iter_cnt == iters_rec[rec_idx]:
            if fh_objgap:
                objgap_abs = fh_objgap(x_new,fworac1)
                objgap_rel = objgap_abs / (-obj_p+objgap_abs) # (p(w)-d(a))/p(w), note that d(a) = -obj_p, and thus (-obj_p+gap_abs) = p(w).
                gaps[rec_idx] = objgap_abs
                if (objgap_abs <= thres_gap2) or (use_rel_gap and objgap_rel <= thres_gap2):
                    finished = True
            elaptime = tm_current - tm_ignore
            fs[rec_idx] = obj_p
            tms[rec_idx] = elaptime
            if not is_stdfw:
                nvertex = len(LacoList)
                vertex[rec_idx] = nvertex
            rec_idx += 1
            if verbose:
                ln_new = 'iter={:05d}, time={:.1f}, '.format(iter_cnt, elaptime)
                ln_new += 'nvertex={}'.format(nvertex) if not is_stdfw else ''
                ln_new += ', obj_p={:g}'.format(obj_p)
                ln_new += ', obj_d={:g}, objgap_abs={:g}, objgap_rel={:g}'.format(obj_p - objgap_abs, objgap_abs, objgap_rel) if fh_objgap else ''
                print(ln_new)
                sys.stdout.flush()
            if finished:
                break
        tm_ignore1 = time.time()
        tm_ignore += tm_ignore1 - tm_ignore0
        
        d_fw = u_old - x_old; 
        if not is_stdfw:
            for alpha, x_pt in LacoList:
                assert(0 <= alpha <= 1)
            
            scos = np.asarray( [ x@grad1 for alpha,x in LacoList ] );
            id_u = np.argmin(scos);
            ud_old = LacoList[id_u][1]
            
            # Update the dictionary.
            if np.linalg.norm(ud_old - u_old) > 1e-10 * nfeas:
                LacoList.append([0.0, u_old])
                i_u = len(LacoList) - 1;
                alph_u = 0; 
            else:
                i_u = id_u;
                alph_u = LacoList[i_u][0];
                u_old = ud_old; 
            
            # Away direction
            if not is_stdfw:
                i_v = np.argmax(scos)
                alph_v, v_old = LacoList[i_v]
        
        # Stopping criteria
        if -d_fw @grad1 <= thres_gap1:
            print('Reached the stopping criteria')
            break
        if is_pfw and i_u == i_v:
            print('Reached PFW stopping criteria')
            break
        
        behave_as_stdfw = False
        if is_afw:
            d_a  = x_old - v_old
            sco_fw = d_fw @ grad1; 
            sco_a  = d_a @ grad1; 
            if sco_fw < sco_a:
                behave_as_stdfw = True
        
        # Line Search
        if is_stdfw or behave_as_stdfw:
            gam_mx = 1.0
            d_old = u_old - x_old
        elif is_pfw:
            gam_mx = min( 1.0 - alph_u, alph_v )
            d_old = u_old - v_old
        elif is_afw:
            gam_mx = np.inf if alph_v == 1.0 else alph_v / (1.0-alph_v)
            d_old = d_a; 
        else:
            assert(0)
        gam_opt = fh_lnsrch( d_old, gam_mx )
        
        # Update
        x_new = x_old + gam_opt * d_old; 
        if not is_stdfw:
            n_alpha = len(LacoList)
            if behave_as_stdfw:
                for i in range(n_alpha):
                    LacoList[i][0] += gam_opt * ( 1.0*(i_u==i) - LacoList[i][0])
            elif is_pfw:
                LacoList[i_u][0] += gam_opt
                LacoList[i_v][0] -= gam_opt
            elif is_afw:
                for i in range(n_alpha):
                    LacoList[i][0] += gam_opt * ( LacoList[i][0] - 1.0*(i_v==i))
            else:
                assert(0)
        
        # Update the dictionary.
        if not is_stdfw:
            thres = 1e-10
            if LacoList[i_v][0] < thres:
                LacoList.pop(i_v)
            elif LacoList[i_u][0] < thres:
                LacoList.pop(i_u)
        
        if iter_cnt == iters_rec[-1]:
            finished = True
            break
    
    valid_idx = (iters_rec <= iter_cnt)
    fs, tms, gaps, iters_rec = [ x[valid_idx] for x in [fs, tms, gaps, iters_rec] ]
    res = {
        'x_est' : x_new,
        'fs' : fs,
        'tms' : tms,
        'gaps' : gaps, 
        'iters_rec' : iters_rec,
        'iter' : iter_cnt,
    }
    if not is_stdfw:
        res['vertex'] = vertex[valid_idx]
    
    return res;

