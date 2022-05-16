import os
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from sklearn import mixture
from joblib import dump, load


# def dpmm_fit(model, train_dataset, args=None, dpmm_components=10, dpmm_downsample_fac=10, pt_dpmm_path=None):
#     """
#     Wrapper for extracting training features for DNS experiment, given a trained DCEC models and a normal training dataset,
#     and fitting a DPMM model
#     :param model: A trained model
#     :param train_dataset: "normal" training dataset, for alpha calculation
#     :param args - command line arguments
#     :param dpmm_components:  Truncation parameter for DPMM
#     :param dpmm_downsample_fac: Downsampling factor for DPMM fitting
#     :param pt_dpmm_path: Path to a pretrained DPMM model
#     :return fitted dpmm model after feature extraction (calc_p)
#     """
#     print("Started fitting DPMM")
#     if pt_dpmm_path is None:
#         # Alpha calculation and fitting
#         train_p = calc_p(model, train_dataset, args, ret_metadata=False)        
#         dpmm_mix = mixture.BayesianGaussianMixture(n_components=dpmm_components,
#                                                    max_iter=500, verbose=1, n_init=1)
#         dpmm_mix.fit(train_p[::dpmm_downsample_fac])

#         try:  # Model persistence
#             dpmm_fn = args.ae_fn.split('.')[0] + '_dpgmm.pkl'
#             dpmm_path = os.path.join(args.ckpt_dir, dpmm_fn)
#             dump(dpmm_mix, dpmm_path)
#         except ModuleNotFoundError:
#             print("Joblib missing, DPMM not saved")
#     else:
#         dpmm_mix = load(pt_dpmm_path)

#     return dpmm_mix


def dpmm_calc_scores(dpmm_mix, model, eval_normal_dataset, args=None):
    """
    Wrapper for extracting evaluation features for DNS experiment, given a trained DCEC model, a trained DPMM model, and two
    datasets for evaluation, a "normal" one and an "abnormal" one
    :dpmm_mix : deep mixture model
    :param model: A trained model    
    :param eval_normal_dataset: "normal" or "mixed" evaluation dataset    
    :param args - command line arguments
    :param ret_metadata:
    :return dpmm scores after feature extraction (calc_p)
    """
    # Alpha calculation and fitting    
    p_vec = calc_p(model, eval_normal_dataset, args)
    dpmm_scores = dpmm_mix.score_samples(p_vec)    
    return dpmm_scores # , eval_normal_dataset.metadata


def calc_p(model, dataset, args):
    """ Evalutates the models output over the data in the dataset. """
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                        shuffle=False, drop_last=False, pin_memory=True)
    model = model.to(args.device)
    model.eval()
    p = p_compute_features(loader, model, device=args.device)

    return p


def p_compute_features(loader, model, device='cuda:0'):
    sfmax = []
    # z_arr = []
    for itern, data_arr in enumerate(loader):
        data = data_arr[0]
        if itern % 100 == 0:
            print("Compute Features Iter {}".format(itern))
        with torch.no_grad():
            data = data.to(device)
            model_ret = model(data, ret_z=False)
            cls_sfmax = model_ret[0]
            cls_sfmax = torch.reshape(cls_sfmax, (cls_sfmax.size(0), -1))
            sfmax.append(cls_sfmax.to('cpu', non_blocking=True).numpy().astype('float32'))

    sfmax = np.concatenate(sfmax)
    return sfmax


def score_dataset(score_vals, metadata, num_frames = None):
    scores_np = get_dataset_scores(score_vals, metadata, num_frames)
    scores_smoothed_np = score_align(scores_np)
    return scores_np, scores_smoothed_np


def get_dataset_scores(scores, metadata, num_frames): #  = None):    
    metadata_np = np.array(metadata)

    # # replace code below to obtain number of frames
    # if num_frames == None:
    #     per_frame_scores_root = 'data/testing/test_frame_mask/'
    #     clip_list = os.listdir(per_frame_scores_root)
    #     clip_list = sorted(fn for fn in clip_list if fn.endswith('.npy'))
    #     clip = clip_list[0]
    #     clip_res_fn = os.path.join(per_frame_scores_root, clip)
    #     clip_gt = np.load(clip_res_fn)
    #     num_frames = clip_gt.shape[0]    

    clip_fig_idxs = set([arr[2] for arr in metadata])
    scores_zeros = np.zeros(num_frames)
    clip_person_scores_dict = {i: np.copy(scores_zeros) for i in clip_fig_idxs}
    for person_id in clip_fig_idxs:
        person_metadata_inds = np.where((metadata_np[:, 2] == person_id))[0]                                        
        pid_scores = scores[person_metadata_inds]
        pid_frame_inds = np.array([metadata[i][3] for i in person_metadata_inds])
        clip_person_scores_dict[person_id][pid_frame_inds] = pid_scores

    clip_ppl_score_arr = np.stack(list(clip_person_scores_dict.values()))
    clip_score = np.amax(clip_ppl_score_arr, axis=0)    

    return clip_score


def score_align(scores_np, seg_len=12, sigma=40):
    scores_shifted = np.zeros_like(scores_np)
    shift = seg_len + (seg_len // 2) - 1
    scores_shifted[shift:] = scores_np[:-shift]
    scores_smoothed = gaussian_filter1d(scores_shifted, sigma)    
    return scores_smoothed


def avg_scores_by_trans(scores, num_transform=5):
    scores_by_trans = scores.reshape(-1, num_transform, order = 'F')
    scores_tavg = scores_by_trans.mean(axis=1)
    return scores_tavg