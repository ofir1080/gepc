import argparse
from re import sub
import subprocess
from os.path import exists
from os.path import join
import logging
import cv2
import numpy as np
from einops import rearrange
from joblib import load
from matplotlib.figure import Figure
import torch
import os
import sys
sys.path.append('/openpose/openpose_vol//gepc/gepc')
import time

# local imports
from defaults import get_cfg_defaults
from inference_utils import *

# gepc utils
from models.dc_gcae.dc_gcae import load_ae_dcec
from utils.pose_seg_dataset import keypoints17_to_coco18
from utils.data_utils import normalize_pose


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

logging.getLogger().setLevel(logging.DEBUG)
parser = argparse.ArgumentParser()

parser.add_argument('--video', default='', type=str, help='Path to video')
parser.add_argument('--num_gpus', default=1, type=int, help='Path to video')
parser.add_argument('--run_local', default=False, type=bool)
# parser.add_argument('--write_json', default=True, type=bool, help='Create JSON output')
args = parser.parse_args()

# parser.add_argument('', default='', type=int, help='')
# parser.add_argument('', default='', type=int, help='')
# parser.add_argument('', default='', type=int, help='')

RUN_LOCAL = args.run_local # id True, a .npy will be read instead of running openpose (since in debug mode the code is excecuted outside of the docker where OpenPose is installed)

def merge_cfg_args(cfg):
    if exists(args.video):
        cfg.VID_IN = args.video
        vid_name = cfg.VID_IN.split('/')[-1]
        # print(vid_name)
        v_name, format = vid_name.split('.')
        cfg.VID_NAME = v_name
        vid_name_out = f'{v_name}_OUT.{format}'
        json_name_out = f'{v_name}_OUT_JSON'
        if cfg.WRITE_VID:
            cfg.VID_OUT = join(cfg.VID_OUT_ROOT, vid_name_out)
        cfg.JSON_OUT = join(cfg.VID_OUT_ROOT, json_name_out)
        assert args.num_gpus > 0, 'Must use at least 1 gpus'
        cfg.OPENPOSE.NUM_GPUS = args.num_gpus
    else:
        raise FileNotFoundError('Video does not exists.')


def create_command(cfg):
    cli_command = ''
    if cfg.POSE_ESTIMATOR == 'openpose':
        assert exists(cfg.OPENPOSE.RUN_FILE)
        cli_command = f'{cfg.OPENPOSE.RUN_FILE} \
                        --video {cfg.VID_IN} \
                        --display 0 \
                        {"--write_video " + cfg.VID_OUT if cfg.WRITE_VID else "--render_pose 0"} \
                        --write_json {cfg.JSON_OUT} \
                        --num_gpu {cfg.OPENPOSE.NUM_GPUS} \
                        {"--disable_blending" if cfg.OPENPOSE.DISABLE_BLENDING and cfg.WRITE_VID else ""}' \
                        # --model_pose {cfg.OPENPOSE.MODEL}'
    elif cfg.POSE_ESTIMATOR == 'alphapose':
        assert exists(cfg.ALPHAPOSE.RUN_FILE), 'Run-file does not exist'
        # os.chdir(cfg.ALPHAPOSE.ROOT)
        cli_command = f'python scripts/demo_inference.py \
                        --cfg {cfg.ALPHAPOSE.CFG_FILE} \
                        --checkpoint {cfg.ALPHAPOSE.CHECKPOINT_PATH} \
                        --video {cfg.VID_IN} \
                        --outdir {cfg.JSON_OUT} \
                        --{cfg.ALPHAPOSE.TRACKER} \
                        --sp'
    return cli_command.split()


def run_pe_inference(pe_root, pe_args):
    if 'openpose' in pe_root:
        subprocess.run(pe_args.command, cwd=pe_root)
    
    elif 'alphapose' in pe_root:
        sys.path.append('/openpose/openpose_vol/Alphapose')
        from alphapose.demo_inference import inference
        inference(**pe_args)


def plot_graphs(plots: dict, vid_out_path):
    logging.getLogger().setLevel(logging.ERROR)

    fg = Figure()
    ax = fg.gca()
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    arange = np.arange(len(plots['per_frame_scores']))
    ax.plot(arange, plots['per_frame_scores'])
    ax.plot(arange, plots['per_frame_scores_smoothed_alighed'])
    fg.savefig(f'{vid_out_path}_plot.png')


def main():
    cfg = get_cfg_defaults()
    merge_cfg_args(cfg)

    if not RUN_LOCAL:
        if cfg.POSE_ESTIMATOR == 'openpose':
            cfg.ROOT = cfg.OPENPOSE.ROOT
            cli_command = create_command(cfg)
            logging.info(f' RUNNING COMMAND:\n\t{" ".join(cli_command)}')
            logging.info(f' Saving frame-wise poses saved in {cfg.JSON_OUT}')
            pe_args = {'command': cli_command}
        elif cfg.POSE_ESTIMATOR == 'alphapose':
            cfg.ROOT = cfg.ALPHAPOSE.ROOT
            pe_args = {
                'video_path': cfg.VID_IN,
                'cfg_file_path': cfg.ALPHAPOSE.CFG_FILE,
                'ckpt_path': cfg.ALPHAPOSE.CHECKPOINT_PATH,
                'outdir_path': cfg.JSON_OUT,
                'tracker': cfg.ALPHAPOSE.TRACKER,
            }

        run_pe_inference(pe=cfg.ROOT, pe_args=pe_args)

        if cfg.POSE_ESTIMATOR == 'openpose':
            # NOTE: task is freezed for now. as of now, the stream outputs a tensor of shape (#person, #frames, 18, 3)
            video_info = json_pack(cfg)
            logging.info(' Finished packing frame-wise jsons')
            video_info_np = prepare_sample(cfg, video_info)
            np.save(join(cfg.VID_OUT_ROOT, cfg.VID_NAME + '_np.npy'), video_info_np)
    else:
        # logging.WARNING('NOTE: eventhough the video is not loaded, it is required for extracting the name')
        video_info_np = np.load(join(cfg.VID_OUT_ROOT_LOCAL, cfg.VID_NAME + '_np.npy'))

    pose_segs_data_np, pose_segs_meta = split_pose_to_segments(video_info_np, seg_stride=1, seg_len=cfg.JSON_FORMATTER.T, scene_id=-1, clip_id=-1)
    
    if cfg.NORM_POSE: # def is True
        # normalize pose
        pose_segs_data_np = normalize_pose(pose_segs_data_np, vid_res=list(get_frame_info(cfg.VID_IN)))

    # convert 17 to 18 (coco convention)
    if cfg.JSON_FORMATTER.V == 17:
        pose_segs_data_np = keypoints17_to_coco18(pose_segs_data_np)

    # filter low scored poses
    # TODO: need to import seg_conf_th_filter. now thresh is 0

    # rearange to shape: TODO
    pose_segs_data_np = np.transpose(pose_segs_data_np, (0, 3, 1, 2)).astype(np.float32)
    if cfg.APPLY_TRANSFORMS:
        # copies every graph and applies cfg.NUM_TRANSFORMS transformation. 
        # pose_segs_data_np = np.tile(pose_segs_data_np, (cfg.NUM_TRANSFORMS,1,1,1))    # shape: [num_graphs*cfg.NUM_TRANSFORMS, 3, 12, 18]
        pose_segs_data_np = apply_transforms(pose_segs_data_np, num_transforms=cfg.NUM_TRANSFORMS)

    pose_segs_data_tn = torch.tensor(pose_segs_data_np)
    pose_segs_data_tn = pose_segs_data_tn.to(device)

    # load pretrained Deep Clustering Graph Convolutional Auto-Encoder model
    if not RUN_LOCAL:
        dc_gcae = load_ae_dcec(cfg.GEPC.DC_GCAE_CKPT)
        dpmm_mix = load(cfg.GEPC.DPMM_CKPT)
    else:
        dc_gcae = load_ae_dcec(cfg.GEPC.DC_GCAE_CKPT_LOCAL)
        dpmm_mix = load(cfg.GEPC.DPMM_CKPT_LOCAL)


    dc_gcae.to(device).eval()
    # feedforward step
    with torch.no_grad():
        cls_sfmax = dc_gcae(pose_segs_data_tn)[0].detach().to('cpu').numpy() # like in p_compute_features func.
    dpmm_scores = dpmm_mix.score_samples(cls_sfmax)

    # takes avg in the transorm dim
    dpmm_scores_avgt = rearrange(dpmm_scores, '(t w) -> t w', t=cfg.NUM_TRANSFORMS) # (cfg.NUM_TRANSFORMS, num_graphs, 3, 12, 18)
    dpmm_scores_avgt = dpmm_scores_avgt.mean(0) # -> (num_graphs, 3, 12, 18)

    num_frames = get_frame_info(cfg.VID_IN, ret_num_frames=True, ret_res=False)
    per_frame_scores = get_per_frame_scores(dpmm_scores_avgt, pose_segs_meta, num_frames=num_frames)
    assert num_frames == len(per_frame_scores)

    per_frame_scores_smoothed_alighed = align_and_smooth_scores(per_frame_scores, seg_len=cfg.JSON_FORMATTER.T, sigma=cfg.SCORE_SMOOTHING_SIGMA)


    plots = {'per_frame_scores': per_frame_scores, 'per_frame_scores_smoothed_alighed': per_frame_scores_smoothed_alighed}
    plot_graphs(plots, join(cfg.VID_OUT_ROOT_LOCAL if RUN_LOCAL else cfg.VID_OUT_ROOT, cfg.VID_NAME))

if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    logging.info(' Overall runtime: {} seconds'.format(end - start))
