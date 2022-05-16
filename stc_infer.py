import os
import random
import collections
import numpy as np
from joblib import load # dump, load

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.gcae.gcae import Encoder
from models.fe.fe_model import init_fenet

from models.dc_gcae.dc_gcae import DC_GCAE, load_ae_dcec
from models.dc_gcae.dc_gcae_training import dc_gcae_train
from models.gcae.gcae_training import Trainer

from utils.data_utils import ae_trans_list
from utils.train_utils import get_fn_suffix, init_clusters
from utils.train_utils import csv_log_dump
from utils.scoring_utils_infer import dpmm_calc_scores, score_dataset, avg_scores_by_trans # , dpmm_fit
from utils.pose_seg_dataset import PoseSegDataset
from utils.pose_ad_argparse import init_stc_parser, init_stc_sub_args
from utils.optim_utils.optim_init import init_optimizer, init_scheduler

import cv2
from matplotlib.figure import Figure


def get_dataset(args):    
    trans_list = ae_trans_list[:args.num_transform]

    dataset_args = {'transform_list': trans_list, 'debug': args.debug, 'headless': args.headless,
                    'scale': args.norm_scale, 'scale_proportional': args.prop_norm_scale, 'seg_len': args.seg_len,
                    'return_indices': True, 'return_metadata': True}
    
    split = 'test'
    dataset_args['seg_stride'] = 1  # No strides for test set
    dataset_args['train_seg_conf_th'] = 0.0
    # dataset = PoseSegDataset(args.pose_path[split], **dataset_args)
    dataset = PoseSegDataset(args.pose_fn, **dataset_args)    
    return dataset


def count_frames_vid(vid_path):
    cap = cv2.VideoCapture(vid_path)
    if cap.isOpened():
        # Joni: may need to read frames manually
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return frame_count
    else:
        raise Exception('Cannot exstract number of frames from {vid_path}')


def count_frames_gt(gt_path):
    clip_gt = np.load(gt_path)
    num_frames = clip_gt.shape[0]
    return num_frames


def count_frames_wrap(path:str):
    if path.endswith('.mp4') or path.endswith('.avi'):
        return count_frames_vid(path)
    elif path.endswith('.npy'):
        return count_frames_gt(path)
    else:
        raise Exception('Cannot exstract number of frames from {path}')


def plot_graphs(plots: dict, vid_out_path):
    import matplotlib.pyplot as plt

    dirname = os.path.dirname(vid_out_path)
    filename = os.path.basename(vid_out_path)

    # plot raw score
    clip_score = plots['per_frame_scores'] # scores_np[start_frames[i]:end_frames[i]]    
    plt.plot(np.arange(len(clip_score)), clip_score, label='Raw Score')
    
    # plot smoothed score    
    scores_smoothed = plots['per_frame_scores_smoothed_alighed']
    plt.plot(np.arange(len(scores_smoothed)), scores_smoothed, 'g', label='Smoothed Score')
        
    plt.legend(loc = 'lower center')    
    plt.xlabel('Frame Index')
    plt.ylabel('DPMM Score')
    plt.title(filename)
    plt.savefig(os.path.join(dirname, filename[:-4] + '.png'))
    # plt.show()


def main():
    parser = init_stc_parser()
    args = parser.parse_args()
    log_dict = collections.defaultdict(int)
    
    args.seed = torch.initial_seed()

    args, ae_args, dcec_args, res_args = init_stc_sub_args(args)
    print(args)
    
    # Load dataset
    dataset = get_dataset(ae_args)
    
    # Load pretrained models    
    dcec_fn = vars(args).get('dcec_fn', None)
    dc_gcae = load_ae_dcec(dcec_fn)
    dc_gcae.eval()    
    pt_dpmm = args.dpmm_fn    
    dpmm_mix = load(pt_dpmm)

    # Calculate normality scores
    dp_scores = dpmm_calc_scores(dpmm_mix, dc_gcae, dataset, args=res_args)
    dp_scores_tavg = avg_scores_by_trans(dp_scores, args.num_transform)
    num_frames = count_frames_wrap(args.vid_fn)
    dp_scores_np, dp_scores_smoothed = score_dataset(dp_scores_tavg, dataset.metadata, num_frames = num_frames)

    # Plot normality scores
    plots = {'per_frame_scores': dp_scores_np, 'per_frame_scores_smoothed_alighed': dp_scores_smoothed}
    # plot_graphs(plots, os.path.dirname(args.vid_fn))
    # plot_graphs(plots, os.path.dirname(args.pose_fn))
    plot_graphs(plots, args.vid_fn)

    # Logging and recording results
    print("Done for {} samples and {} trans".format(dp_scores_tavg.shape[0], args.num_transform))
    # csv_log_dump(args, log_dict)


if __name__ == '__main__':
    main()