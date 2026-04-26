import numpy as np
from scipy.optimize import linear_sum_assignment

def get_bounding_box(img):
    """Get bounding box coordinate information."""
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]

def remap_label(pred, by_size=False):
    """
    Rename all instance id so that the id is contiguous i.e [0, 1, 2, 3]
    not [0, 2, 4, 6]. The ordering of instances (which one comes first)
    is preserved unless by_size=True, then the instances will be reordered
    so that bigger nucler has smaller ID

    Args:
        pred    : the 2d array contain instances where each instances is marked
                  by non-zero integer
        by_size : renaming with larger nuclei has smaller id (on-top)
    """
    pred_id = list(np.unique(pred))
    if 0 in pred_id:
        pred_id.remove(0)
    if len(pred_id) == 0:
        return pred  # no label
    if by_size:
        pred_size = []
        for inst_id in pred_id:
            size = (pred == inst_id).sum()
            pred_size.append(size)
        # sort the id by size in descending order
        pair_list = zip(pred_id, pred_size)
        pair_list = sorted(pair_list, key=lambda x: x[1], reverse=True)
        pred_id, pred_size = zip(*pair_list)

    new_pred = np.zeros(pred.shape, np.int32)
    for idx, inst_id in enumerate(pred_id):
        new_pred[pred == inst_id] = idx + 1
    return new_pred

def cell_detection_scores(paired_true, paired_pred, unpaired_true, unpaired_pred, w=[1, 1]):
    tp_d = paired_pred.shape[0]
    fp_d = unpaired_pred.shape[0]
    fn_d = unpaired_true.shape[0]
    prec_d = tp_d / (tp_d + fp_d) if (tp_d + fp_d) > 0 else 0.0
    rec_d  = tp_d / (tp_d + fn_d) if (tp_d + fn_d) > 0 else 0.0
    denom = (2 * tp_d + w[0] * fp_d + w[1] * fn_d)
    f1_d = (2 * tp_d / denom) if denom > 0 else 0.0
    return float(f1_d), float(prec_d), float(rec_d)

def get_dice_1(true, pred):
    true = np.copy(true)
    pred = np.copy(pred)
    true[true > 0] = 1
    pred[pred > 0] = 1
    inter = true * pred
    denom = true + pred
    return float(2.0 * np.sum(inter) / (np.sum(denom) + 1.0e-6))
"""

def get_pq(true, pred, match_iou=0.5, remap=True):
    \"""Get the panoptic quality result.

    Fast computation requires instance IDs are in contiguous orderding i.e [1, 2, 3, 4]
    not [2, 3, 6, 10]. Please call `remap_label` beforehand. Here, the `by_size` flag
    has no effect on the result.

    Args:
        true (ndarray): HxW ground truth instance segmentation map
        pred (ndarray): HxW predicted instance segmentation map
        match_iou (float): IoU threshold level to determine the pairing between
            GT instances `p` and prediction instances `g`. `p` and `g` is a pair
            if IoU > `match_iou`. However, pair of `p` and `g` must be unique
            (1 prediction instance to 1 GT instance mapping). If `match_iou` < 0.5,
            Munkres assignment (solving minimum weight matching in bipartite graphs)
            is caculated to find the maximal amount of unique pairing. If
            `match_iou` >= 0.5, all IoU(p,g) > 0.5 pairing is proven to be unique and
            the number of pairs is also maximal.
        remap (bool): whether to ensure contiguous ordering of instances.

    Returns:
        [dq, sq, pq]: measurement statistic

        [paired_true, paired_pred, unpaired_true, unpaired_pred]:
                      pairing information to perform measurement

        paired_iou.sum(): sum of IoU within true positive predictions

    \"""
    assert match_iou >= 0.0, "Cant' be negative"
    # ensure instance maps are contiguous
    if remap:
        pred = remap_label(pred)
        true = remap_label(true)

    true = np.copy(true)
    pred = np.copy(pred)
    true = true.astype("int32")
    pred = pred.astype("int32")
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))
    # prefill with value
    pairwise_iou = np.zeros([len(true_id_list), len(pred_id_list)], dtype=np.float64)

    # caching pairwise iou
    for true_id in true_id_list[1:]:  # 0-th is background
        t_mask_lab = true == true_id
        rmin1, rmax1, cmin1, cmax1 = get_bounding_box(t_mask_lab)
        t_mask_crop = t_mask_lab[rmin1:rmax1, cmin1:cmax1]
        t_mask_crop = t_mask_crop.astype("int")
        p_mask_crop = pred[rmin1:rmax1, cmin1:cmax1]
        pred_true_overlap = p_mask_crop[t_mask_crop > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        for pred_id in pred_true_overlap_id:
            if pred_id == 0:  # ignore
                continue  # overlaping background
            p_mask_lab = pred == pred_id
            p_mask_lab = p_mask_lab.astype("int")

            # crop region to speed up computation
            rmin2, rmax2, cmin2, cmax2 = get_bounding_box(p_mask_lab)
            rmin = min(rmin1, rmin2)
            rmax = max(rmax1, rmax2)
            cmin = min(cmin1, cmin2)
            cmax = max(cmax1, cmax2)
            t_mask_crop2 = t_mask_lab[rmin:rmax, cmin:cmax]
            p_mask_crop2 = p_mask_lab[rmin:rmax, cmin:cmax]

            total = (t_mask_crop2 + p_mask_crop2).sum()
            inter = (t_mask_crop2 * p_mask_crop2).sum()
            iou = inter / (total - inter)
            pairwise_iou[true_id - 1, pred_id - 1] = iou

    if match_iou >= 0.5:
        paired_iou = pairwise_iou[pairwise_iou > match_iou]
        pairwise_iou[pairwise_iou <= match_iou] = 0.0
        paired_true, paired_pred = np.nonzero(pairwise_iou)
        paired_iou = pairwise_iou[paired_true, paired_pred]
        paired_true += 1  # index is instance id - 1
        paired_pred += 1  # hence return back to original
    else:  # * Exhaustive maximal unique pairing
        #### Munkres pairing with scipy library
        # the algorithm return (row indices, matched column indices)
        # if there is multiple same cost in a row, index of first occurence
        # is return, thus the unique pairing is ensure
        # inverse pair to get high IoU as minimum
        paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)
        ### extract the paired cost and remove invalid pair
        paired_iou = pairwise_iou[paired_true, paired_pred]

        # now select those above threshold level
        # paired with iou = 0.0 i.e no intersection => FP or FN
        paired_true = list(paired_true[paired_iou > match_iou] + 1)
        paired_pred = list(paired_pred[paired_iou > match_iou] + 1)
        paired_iou = paired_iou[paired_iou > match_iou]

    # get the actual FP and FN
    unpaired_true = [idx for idx in true_id_list[1:] if idx not in paired_true]
    unpaired_pred = [idx for idx in pred_id_list[1:] if idx not in paired_pred]
    # print(paired_iou.shape, paired_true.shape, len(unpaired_true), len(unpaired_pred))

    #
    tp = len(paired_true)
    fp = len(unpaired_pred)
    fn = len(unpaired_true)
    # get the F1-score i.e DQ
    dq = tp / ((tp + 0.5 * fp + 0.5 * fn) + 1.0e-6)
    # get the SQ, no paired has 0 iou so not impact
    sq = paired_iou.sum() / (tp + 1.0e-6)

    return (
        [dq, sq, dq * sq],
        [tp, fp, fn],
        [np.array(paired_true), np.array(paired_pred), np.array(unpaired_pred), np.array(unpaired_true)],
        paired_iou.sum(),
    )
"""


def get_pq(true, pred, match_iou=0.5, remap=True):
    assert match_iou >= 0.0, "Cant' be negative"
    
    # if remap:
    #     pred = remap_label(pred)
    #     true = remap_label(true)

    true = true.astype(np.int32)
    pred = pred.astype(np.int32)
    
    max_true = int(true.max())
    max_pred = int(pred.max())
    
    # 1. Calculer l'aire (nombre de pixels) de chaque instance en 1 seul passage !
    true_areas = np.bincount(true.ravel())
    pred_areas = np.bincount(pred.ravel())

    # 2. Trouver les intersections entre TOUS les ground truths et TOUTES les prédictions
    # Astuce mathématique : on combine les deux images en une seule équation
    offset = max_pred + 1
    combo = (true.astype(np.int64) * offset) + pred.astype(np.int64)
    
    # bincount compte combien de fois chaque combinaison (vrai_noyau, pred_noyau) apparaît
    combo_counts = np.bincount(combo.ravel())
    
    # 3. Extraire les paires qui se chevauchent vraiment
    non_zero_indices = np.nonzero(combo_counts)[0]
    t_ids = non_zero_indices // offset
    p_ids = non_zero_indices % offset
    intersections = combo_counts[non_zero_indices]

    # 4. Remplir la matrice d'IoU (Instantanné car on ne boucle que sur les vrais chevauchements)
    pairwise_iou = np.zeros([max_true, max_pred], dtype=np.float64)

    for t_id, p_id, inter in zip(t_ids, p_ids, intersections):
        if t_id == 0 or p_id == 0:
            continue # On ignore le fond (background)
            
        union = true_areas[t_id] + pred_areas[p_id] - inter
        pairwise_iou[t_id - 1, p_id - 1] = inter / union

    # --- LE RESTE DU CODE RESTE IDENTIQUE ---
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))

    if match_iou >= 0.5:
        paired_iou = pairwise_iou[pairwise_iou > match_iou]
        pairwise_iou[pairwise_iou <= match_iou] = 0.0
        paired_true, paired_pred = np.nonzero(pairwise_iou)
        paired_iou = pairwise_iou[paired_true, paired_pred]
        paired_true += 1  
        paired_pred += 1  
    else:  
        paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)
        paired_iou = pairwise_iou[paired_true, paired_pred]

        paired_true = list(paired_true[paired_iou > match_iou] + 1)
        paired_pred = list(paired_pred[paired_iou > match_iou] + 1)
        paired_iou = paired_iou[paired_iou > match_iou]

    unpaired_true = [idx for idx in true_id_list[1:] if idx not in paired_true]
    unpaired_pred = [idx for idx in pred_id_list[1:] if idx not in paired_pred]

    tp = len(paired_true)
    fp = len(unpaired_pred)
    fn = len(unpaired_true)
    
    dq = tp / ((tp + 0.5 * fp + 0.5 * fn) + 1.0e-6)
    sq = paired_iou.sum() / (tp + 1.0e-6)

    return (
        [dq, sq, dq * sq],
        [tp, fp, fn],
        [np.array(paired_true), np.array(paired_pred), np.array(unpaired_pred), np.array(unpaired_true)],
        paired_iou.sum(),
    )