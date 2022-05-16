from yacs.config import CfgNode as CN
from os.path import join


VOLUME_DIR_PATH = '/openpose/openpose_vol'
LOCAL_DIR_PATH = '/datasas/iic_projects/magnetons/tau_anomaly_detection/openpose_vol'

_C = CN()

_C.POSE_ESTIMATOR = 'alphapose' # {openpose, alphapose}

_C.WRITE_VID = False
_C.VID_OUT_ROOT = join(VOLUME_DIR_PATH, 'OUTPUTS')
_C.VID_OUT_ROOT_LOCAL = join(LOCAL_DIR_PATH, 'OUTPUTS')

_C.VID_IN = ''
_C.VID_OUT = './OUT.mp4'

_C.JSON_OUT = join(VOLUME_DIR_PATH, 'OUTPUTS', _C.VID_IN + '_JSON_OUT.json')


#################
# OpenPose args #
#################


_C.OPENPOSE = CN()

_C.OPENPOSE.ROOT = '/openpose/'
_C.OPENPOSE.RUN_FILE = join(_C.OPENPOSE.ROOT, '/openpose/build/examples/openpose/openpose.bin')
_C.OPENPOSE.KP_FORMAT = 'COCO' # {COCO, BODY25} 

# num gpus to use while extracting poses
_C.OPENPOSE.NUM_GPUS = 2

# display setting
_C.OPENPOSE.DISPLAY = False
_C.OPENPOSE.DISABLE_BLENDING = False

_C.OPENPOSE.MODEL = 'BODY_25' # {COCO,BODY_25,MPI,MPI_4_layers}


##################
# AlphaPose args #
##################

_C.ALPHAPOSE = CN()

_C.ALPHAPOSE.ROOT = '/alphapose/AlphaPose'
_C.ALPHAPOSE.RUN_FILE = join(_C.ALPHAPOSE.ROOT, 'scripts/demo_inference.py')

_C.ALPHAPOSE.CFG_FILE = join(_C.ALPHAPOSE.ROOT, 'configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml')
_C.ALPHAPOSE.TRACKER = 'pose_track' # {pose_track, pose_flow, detector tracker}
_C.ALPHAPOSE.SAVE_VID = False

_C.ALPHAPOSE.CHECKPOINT_PATH = join(_C.ALPHAPOSE.ROOT, 'pretrained_models/fast_res50_256x192.pth')

##################


##################
# GEPC args #
##################

_C.GEPC = CN()

_C.GEPC.DC_GCAE_CKPT = join(VOLUME_DIR_PATH, 'gepc/gepc/data/exp_dir/stc/Jan23_1249/checkpoints/Jan23_1250_stc_sagc_checkpoint/done25_Jan23_1434_dcec10_06_checkpoint.pth.tar')
_C.GEPC.DC_GCAE_CKPT_LOCAL = join(LOCAL_DIR_PATH, 'gepc/gepc/data/exp_dir/stc/Jan23_1249/checkpoints/Jan23_1250_stc_sagc_checkpoint/done25_Jan23_1434_dcec10_06_checkpoint.pth.tar')

_C.GEPC.DPMM_CKPT = join(VOLUME_DIR_PATH, 'gepc/gepc/data/exp_dir/stc/Jan23_1249/checkpoints/Jan23_1250_stc_sagc_checkpoint_dpgmm.pkl')
_C.GEPC.DPMM_CKPT_LOCAL = join(LOCAL_DIR_PATH, 'gepc/gepc/data/exp_dir/stc/Jan23_1249/checkpoints/Jan23_1250_stc_sagc_checkpoint_dpgmm.pkl')

# for running local (at zil07)
# _C.GEPC.DC_GCAE_CKPT = '/datasas/iic_projects/magnetons/tau_anomaly_detection/openpose_vol/gepc/gepc/data/exp_dir/stc/Jan23_1249/checkpoints/Jan23_1250_stc_sagc_checkpoint/done25_Jan23_1434_dcec10_06_checkpoint.pth.tar'
# _C.GEPC.DPMM_CKPT = '/datasas/iic_projects/magnetons/tau_anomaly_detection/openpose_vol/gepc/gepc/data/exp_dir/stc/Jan23_1249/checkpoints/Jan23_1250_stc_sagc_checkpoint_dpgmm.pkl'


_C.JSON_FORMATTER = CN()

_C.JSON_FORMATTER.F_WIDTH = 856   # SHEN TECH
_C.JSON_FORMATTER.F_HEIGHT = 480  # SHEN TECH

_C.JSON_FORMATTER.C = 3   # channel
_C.JSON_FORMATTER.T = 12 # frame
_C.JSON_FORMATTER.V = 18  # joint
_C.JSON_FORMATTER.M = 10   # person
# _C.JSON_FORMATTER

_C.MATCH_POSE = True # if ture, match the pose between two frames
_C.NORM_POSE = True
_C.NUM_TRANSFORMS = 5
_C.APPLY_TRANSFORMS = True
_C.SCORE_SMOOTHING_SIGMA = 40


def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()







