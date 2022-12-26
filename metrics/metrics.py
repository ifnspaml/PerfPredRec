import torch
import numpy as np
import warnings

def compute_psnr(img, gen_img):
    # Function for computing the peak signal-tp-noise ratio (PSNR)
    # img: original image; gen_img: generated image

    # Compute MSE first
    mse = ((img - gen_img) ** 2).mean()

    # Compute PSNR with MSE
    psnr = 10 * torch.log10(1 / mse)

    return psnr

def fast_hist(gt, pred, n_class):
    mask_true = (gt >= 0) & (gt < n_class)
    mask_pred = (pred >= 0) & (pred < n_class)
    mask = mask_pred & mask_true
    label_true = gt[mask].astype(np.int)
    label_pred = pred[mask].astype(np.int)
    hist = np.bincount(n_class * label_true + label_pred,
                       minlength=n_class * n_class).reshape(n_class, n_class).astype(np.float)
    return hist

def compute_miou(gt, pred, n_class):
    # Function for computing the peak signal-tp-noise ratio (PSNR)
    # img: original image; gen_img: generated image
    confusion_matrix=np.zeros((n_class, n_class))
    gt=np.array(gt.cpu())
    pred=np.array(pred.cpu())
    for lt, lp in zip(gt, pred):
        hist=fast_hist(lt.flatten(), lp.flatten(), n_class)
        confusion_matrix += hist
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        iou = np.diag(confusion_matrix) / \
             (confusion_matrix.sum(axis=1) + confusion_matrix.sum(axis=0) - np.diag(confusion_matrix))
    meaniou = np.nanmean(iou)
    return meaniou

def transform2array(metrics):
    """ Transform all metrics entries to numpy arrays if necessary """
    for c in metrics.keys():
        for s in metrics[c].keys():
            for m in metrics[c][s].keys():
                if isinstance(metrics[c][s][m], str):
                    metrics[c][s][m] = np.fromstring(metrics[c][s][m], sep=',')
                else:
                    pass
                if 'img_miou' in m:
                    metrics[c][s][m] = metrics[c][s][m] * 100
    return metrics
