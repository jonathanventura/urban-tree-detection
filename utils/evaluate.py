import numpy as np

from skimage.feature import peak_local_max
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import linear_sum_assignment

from matplotlib import pyplot as plt

import tqdm

def find_matching(gt_indices,pred_indices,max_distance):
    if len(gt_indices)==0 or len(pred_indices)==0:
        dists = np.ones((len(gt_indices),len(pred_indices)),dtype='float32')*np.inf
    else:
        # calculate pairwise distances
        dists = pairwise_distances(gt_indices,pred_indices)
    
        # associate each gt tree with all pred trees within radius
        dists[dists>max_distance] = np.inf

    # find optimal assignment
    maxval = 1e9
    cost_matrix = np.copy(dists)
    cost_matrix[np.isinf(cost_matrix)] = maxval
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    dists[:] = np.inf
    dists[row_ind,col_ind] = cost_matrix[row_ind,col_ind]
    dists[dists>=maxval] = np.inf
    
    # associated pred trees = true positives
    #assert(np.max(np.sum(~np.isinf(dists),axis=0))<=1)
    #assert(np.max(np.sum(~np.isinf(dists),axis=1))<=1)
    assoc = np.where(~np.isinf(dists))
    tp_gt_inds = assoc[0]
    tp_inds = assoc[1]
    tp = len(tp_inds)

    # un-associated pred trees = false positives
    fp_inds = np.where(np.all(np.isinf(dists),axis=0))[0]
    fp = len(fp_inds)

    # un-associated gt trees = false negatives
    fn_inds = np.where(np.all(np.isinf(dists),axis=1))[0]
    fn = len(fn_inds)
    
    if dists[:,tp_inds].size>0:
        tp_dists = np.min(dists[:,tp_inds],axis=0)
    else:
        tp_dists = []
    
    return tp, fp, fn, tp_dists

def test_all_thresholds(gts, preds, max_distance):
    all_gt_indices = []
    all_pred_indices = []
    all_pred_abs = []
    all_pred_rel = []

    for i in range(len(preds)):
        gt = gts[i]
        pred = preds[i]
        
        gt_rows, gt_cols = np.where(gt>0)
        gt_indices = np.stack([gt_rows,gt_cols],axis=-1)
        pred_indices = peak_local_max(pred,min_distance=1,threshold_abs=0,threshold_rel=None)
        pred_abs = pred[pred_indices[:,0],pred_indices[:,1]]
        pred_rel = pred_abs/pred.max()
        
        all_gt_indices.append(gt_indices)
        all_pred_indices.append(pred_indices)
        all_pred_abs.append(pred_abs)
        all_pred_rel.append(pred_rel)
    
    pred_abs_sorted = sorted(np.concatenate(all_pred_abs,axis=0).flatten(),reverse=True)
    pred_rel_sorted = sorted(np.concatenate(all_pred_rel,axis=0).flatten(),reverse=True)
    
    thresholds = []
    precisions = []
    recalls = []
    
    pbar = tqdm.tqdm(total=len(pred_abs_sorted))
    for i,thresh in enumerate(pred_abs_sorted):
        my_pred_indices = [pred_indices[pred_abs>=thresh] for pred_indices,pred_abs in zip(all_pred_indices,all_pred_abs)]
    #pbar = tqdm.tqdm(total=len(pred_rel_sorted))
    #for i,thresh in enumerate(pred_rel_sorted):
        #my_pred_indices = [pred_indices[pred_rel>=thresh] for pred_indices,pred_rel in zip(all_pred_indices,all_pred_rel)]

        all_tp = 0
        all_fp = 0
        all_fn = 0
        
        for gt_indices, pred_indices in zip(all_gt_indices,my_pred_indices):
            tp, fp, fn, _ = find_matching(gt_indices,pred_indices,max_distance)
            all_tp += tp
            all_fp += fp
            all_fn += fn
        
        precision = all_tp/(all_tp+all_fp) if all_tp+all_fp>0 else 0
        recall = all_tp/(all_tp+all_fn) if all_tp+all_fn>0 else 0
        
        thresholds.append(thresh)
        precisions.append(precision)
        recalls.append(recall)
        
        pbar.update(1)

    thresholds = np.array(thresholds)
    precisions = np.array(precisions)
    recalls = np.array(recalls)

    return thresholds, precisions, recalls

def calculate_ap(precisions,recalls):
    return np.sum((recalls[1:]-recalls[:-1])*precisions[1:])

def evaluate(gts, preds, min_distance, threshold_rel, threshold_abs, max_distance, return_locs=False):
    """ Evaluate precision/recall metrics on prediction.
        Arguments:
            gts: ground truth annotation (0 = non-tree, 1 = tree) [N,H,W] 
            preds: predicted confidence maps [N,H,W]
            min_distance: minimum distance between detections
            threshold_rel: relative threshold for local peak finding (None to disable)
            threshold_abs: absolute threshold for local peak finding (None to disable)
            max_distance: maximum distance from detection to gt point 
            return_locs: whether to return the locations of true positives, false positives, etc.
        Returns:
            Result dictionary containing precision, recall, F-score, and RMSE metrics.
            If return_locs = True, the following extra information will be included in the dictionary:
                tp_locs: x,y locations of true positives
                tp_gt_locs: x,y locations of ground truth points associated with true positives
                fp_locs: x,y locations of false positives
                fn_locs: x,y locations of false negatives
                gt_locs: x,y locations of ground truth points
    """
    all_tp = 0
    all_fp = 0
    all_fn = 0
    all_tp_dists = []
    
    if return_locs:
        all_tp_locs = []
        all_tp_gt_locs = []
        all_fp_locs = []
        all_fn_locs = []
        all_gt_locs = []

    for gt,pred in zip(gts,preds):
        gt_rows, gt_cols = np.where(gt>0)
        gt_indices = np.stack([gt_rows,gt_cols],axis=-1)
        pred_indices = peak_local_max(pred,min_distance=min_distance,threshold_abs=threshold_abs,threshold_rel=threshold_rel)

        if len(gt_indices)==0 or len(pred_indices)==0:
            dists = np.ones((len(gt_indices),len(pred_indices)),dtype='float32')*np.inf
        else:
            # calculate pairwise distances
            dists = pairwise_distances(gt_indices,pred_indices)
        
            # associate each gt tree with all pred trees within radius
            dists[dists>max_distance] = np.inf

        # find optimal assignment
        maxval = 1e9
        cost_matrix = np.copy(dists)
        cost_matrix[np.isinf(cost_matrix)] = maxval
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        dists[:] = np.inf
        dists[row_ind,col_ind] = cost_matrix[row_ind,col_ind]
        dists[dists>=maxval] = np.inf
        
        # associated pred trees = true positives
        assoc = np.where(~np.isinf(dists))
        tp_gt_inds = assoc[0]
        tp_inds = assoc[1]
        tp = len(tp_inds)

        # un-associated pred trees = false positives
        fp_inds = np.where(np.all(np.isinf(dists),axis=0))[0]
        fp = len(fp_inds)

        # un-associated gt trees = false negatives
        fn_inds = np.where(np.all(np.isinf(dists),axis=1))[0]
        fn = len(fn_inds)
        
        if dists[:,tp_inds].size>0:
            tp_dists = np.min(dists[:,tp_inds],axis=0)
        else:
            tp_dists = []
        
        all_tp += tp
        all_fp += fp
        all_fn += fn
        all_tp_dists.append(tp_dists)
    
        if return_locs:
            tp_locs = []
            tp_gt_locs = []
            fp_locs = []
            fn_locs = []
            gt_locs = []

            for y,x in gt_indices:
                gt_locs.append([x,y])
            for y,x in gt_indices[fn_inds]:
                fn_locs.append([x,y])
            for (y,x),(gty,gtx) in zip(pred_indices[tp_inds],
                                       gt_indices[tp_gt_inds]):
                tp_locs.append([x,y])
                tp_gt_locs.append([gtx,gty])
            for y,x in pred_indices[fp_inds]:
                fp_locs.append([x,y])

            tp_locs = np.array(tp_locs)
            tp_gt_locs = np.array(tp_gt_locs)
            fp_locs = np.array(fp_locs)
            fn_locs = np.array(fn_locs)
            gt_locs = np.array(gt_locs)

            all_tp_locs.append(tp_locs)
            all_tp_gt_locs.append(tp_gt_locs)
            all_fp_locs.append(fp_locs)
            all_fn_locs.append(fn_locs)
            all_gt_locs.append(gt_locs)
    
    all_tp_dists = np.concatenate(all_tp_dists)

    precision = all_tp/(all_tp+all_fp) if all_tp+all_fp>0 else 0
    recall = all_tp/(all_tp+all_fn) if all_tp+all_fn>0 else 0
    fscore = 2*(precision*recall)/(precision+recall) if precision+recall>0 else 0
    rmse = np.sqrt(np.mean(all_tp_dists**2)) if len(all_tp_dists)>0 else np.inf 
    
    results = {
        'precision':precision,
        'recall':recall,
        'fscore':fscore,
        'rmse':rmse,
    }
    if return_locs:
        results.update({
            'tp_locs':all_tp_locs,
            'tp_gt_locs':all_tp_gt_locs,
            'fp_locs':all_fp_locs,
            'fn_locs':all_fn_locs,
            'gt_locs':all_gt_locs,
        })
    return results

def make_figure(images,results,num_cols=5):
    num_rows = len(images)//num_cols+1
    fig,ax = plt.subplots(num_rows,num_cols,figsize=(8.5,11),tight_layout=True)
    for a in ax.flatten(): a.axis('off')
    tp_locs = results['tp_locs']
    tp_gt_locs = results['tp_gt_locs']
    fp_locs = results['fp_locs']
    fn_locs = results['fn_locs']
    gt_locs = results['gt_locs']
    for a,im,tp,tp_gt,fp,fn,gt in zip(ax.flatten(),images,
                        tp_locs,tp_gt_locs,fp_locs,fn_locs,gt_locs):
        a.imshow(im)

        if len(gt)>0:
            if len(gt.shape)==1: gt = gt[None,:]
            a.plot(gt[:,0],gt[:,1],'m.')
        
        if len(tp)>0:
            if len(tp.shape)==1: tp = tp[None,:]
            a.plot(tp[:,0],tp[:,1],'g+')

        if len(fp)>0:
            if len(fp.shape)==1: fp = fp[None,:]
            a.plot(fp[:,0],fp[:,1],'y^')

        if len(fn)>0:
            if len(fn.shape)==1: fn = fn[None,:]
            a.plot(fn[:,0],fn[:,1],'m.',markeredgecolor='k',markeredgewidth=1)
        
        if len(tp_gt)>0:
            if len(tp_gt.shape)==1: tp_gt= tp_gt[None,:]
            for t,g in zip(tp,tp_gt):
                a.plot((t[0],g[0]),(t[1],g[1]),'y-')
        
    return fig

