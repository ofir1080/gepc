
import numpy as np
import random
from pathlib import Path
import json
import logging
import cv2
from einops import rearrange
from scipy.ndimage import gaussian_filter1d
import sys
sys.path.insert(0, '/openpose/openpose_vol//gepc/gepc')

# local imports
from utils.data_utils import PoseTransform


##################
# OpenPose utils #
##################

###################################################
# Borrowed from https://github.com/yysijie/st-gcn #
###################################################


def downsample(data_numpy, step, random_sample=True):
    # input: C,T,V,M
    begin = np.random.randint(step) if random_sample else 0
    return data_numpy[:, begin::step, :, :]


def temporal_slice(data_numpy, step):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    return data_numpy.reshape(C, T / step, step, V, M).transpose(
        (0, 1, 3, 2, 4)).reshape(C, T / step, V, step * M)


def mean_subtractor(data_numpy, mean):
    # input: C,T,V,M
    # naive version
    if mean == 0:
        return
    C, T, V, M = data_numpy.shape
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()
    data_numpy[:, :end, :, :] = data_numpy[:, :end, :, :] - mean
    return data_numpy


def auto_pading(data_numpy, size, random_pad=False):
    C, T, V, M = data_numpy.shape
    if T < size:
        begin = random.randint(0, size - T) if random_pad else 0
        data_numpy_paded = np.zeros((C, size, V, M))
        data_numpy_paded[:, begin:begin + T, :, :] = data_numpy
        return data_numpy_paded
    else:
        return data_numpy


def random_choose(data_numpy, size, auto_pad=True):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    if T == size:
        return data_numpy
    elif T < size:
        if auto_pad:
            return auto_pading(data_numpy, size, random_pad=True)
        else:
            return data_numpy
    else:
        begin = random.randint(0, T - size)
        return data_numpy[:, begin:begin + size, :, :]


def random_move(data_numpy,
                angle_candidate=[-10., -5., 0., 5., 10.],
                scale_candidate=[0.9, 1.0, 1.1],
                transform_candidate=[-0.2, -0.1, 0.0, 0.1, 0.2],
                move_time_candidate=[1]):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    move_time = random.choice(move_time_candidate)
    node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
    node = np.append(node, T)
    num_node = len(node)

    A = np.random.choice(angle_candidate, num_node)
    S = np.random.choice(scale_candidate, num_node)
    T_x = np.random.choice(transform_candidate, num_node)
    T_y = np.random.choice(transform_candidate, num_node)

    a = np.zeros(T)
    s = np.zeros(T)
    t_x = np.zeros(T)
    t_y = np.zeros(T)

    # linspace
    for i in range(num_node - 1):
        a[node[i]:node[i + 1]] = np.linspace(
            A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
        s[node[i]:node[i + 1]] = np.linspace(S[i], S[i + 1],
                                             node[i + 1] - node[i])
        t_x[node[i]:node[i + 1]] = np.linspace(T_x[i], T_x[i + 1],
                                               node[i + 1] - node[i])
        t_y[node[i]:node[i + 1]] = np.linspace(T_y[i], T_y[i + 1],
                                               node[i + 1] - node[i])

    theta = np.array([[np.cos(a) * s, -np.sin(a) * s],
                      [np.sin(a) * s, np.cos(a) * s]])

    # perform transformation
    for i_frame in range(T):
        xy = data_numpy[0:2, i_frame, :, :]
        new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1))
        new_xy[0] += t_x[i_frame]
        new_xy[1] += t_y[i_frame]
        data_numpy[0:2, i_frame, :, :] = new_xy.reshape(2, V, M)

    return data_numpy


def random_shift(data_numpy):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    data_shift = np.zeros(data_numpy.shape)
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()

    size = end - begin
    bias = random.randint(0, T - size)
    data_shift[:, bias:bias + size, :, :] = data_numpy[:, begin:end, :, :]

    return data_shift


def openpose_match(data_numpy):
    C, T, V, M = data_numpy.shape
    assert (C == 3)
    score = data_numpy[2, :, :, :].sum(axis=1)
    # the rank of body confidence in each frame (shape: T-1, M)
    rank = (-score[0:T - 1]).argsort(axis=1).reshape(T - 1, M)

    # data of frame 1
    xy1 = data_numpy[0:2, 0:T - 1, :, :].reshape(2, T - 1, V, M, 1)
    # data of frame 2
    xy2 = data_numpy[0:2, 1:T, :, :].reshape(2, T - 1, V, 1, M)
    # square of distance between frame 1&2 (shape: T-1, M, M)
    distance = ((xy2 - xy1)**2).sum(axis=2).sum(axis=0)

    # match pose
    forward_map = np.zeros((T, M), dtype=int) - 1
    forward_map[0] = range(M)
    for m in range(M):
        choose = (rank == m)
        forward = distance[choose].argmin(axis=1)
        for t in range(T - 1):
            distance[t, :, forward[t]] = np.inf
        forward_map[1:][choose] = forward
    assert (np.all(forward_map >= 0))

    # string data
    for t in range(T - 1):
        forward_map[t + 1] = forward_map[t + 1][forward_map[t]]

    # generate data
    new_data_numpy = np.zeros(data_numpy.shape)
    for t in range(T):
        new_data_numpy[:, t, :, :] = data_numpy[:, t, :, forward_map[
            t]].transpose(1, 2, 0)
    data_numpy = new_data_numpy

    # score sort
    trace_score = data_numpy[2, :, :, :].sum(axis=1).sum(axis=0)
    rank = (-trace_score).argsort()
    data_numpy = data_numpy[:, :, :, rank]

    return data_numpy


def top_k_by_category(label, score, top_k):
    instance_num, class_num = score.shape
    rank = score.argsort()
    hit_top_k = [[] for i in range(class_num)]
    for i in range(instance_num):
        l = label[i]
        hit_top_k[l].append(l in rank[i, -top_k:])

    accuracy_list = []
    for hit_per_category in hit_top_k:
        if hit_per_category:
            accuracy_list.append(sum(hit_per_category) * 1.0 / len(hit_per_category))
        else:
            accuracy_list.append(0.0)
    return accuracy_list


def calculate_recall_precision(label, score):
    instance_num, class_num = score.shape
    rank = score.argsort()
    confusion_matrix = np.zeros([class_num, class_num])

    for i in range(instance_num):
        true_l = label[i]
        pred_l = rank[i, -1]
        confusion_matrix[true_l][pred_l] += 1

    precision = []
    recall = []

    for i in range(class_num):
        true_p = confusion_matrix[i][i]
        false_n = sum(confusion_matrix[i, :]) - true_p
        false_p = sum(confusion_matrix[:, i]) - true_p
        precision.append(true_p * 1.0 / (true_p + false_p))
        recall.append(true_p * 1.0 / (true_p + false_n))

    return precision, recall


def json_pack(cfg): # , video_name, frame_width, frame_height, label='unknown', label_index=-1):
    sequence_info = []
    p = Path(cfg.JSON_OUT)
    video_name = Path(cfg.VID_IN).stem
    # video resolution
    w, h = get_frame_info(cfg.VID_IN)
    logging.info(f' Video resultion: {w} x {h}')

    for path in p.glob(video_name + '*.json'):
        json_path = str(path)
        # print(path)
        frame_id = int(path.stem.split('_')[-2])
        frame_data = {'frame_index': frame_id}
        data = json.load(open(json_path))
        skeletons = []
        for person in data['people']:
            score, coordinates = [], []
            skeleton = {}
            keypoints = person['pose_keypoints_2d']
            for i in range(0, len(keypoints), 3):
                if i == 8*3 or i > 18*3:
                    continue
                # coordinates += [keypoints[i] / w, keypoints[i + 1] / h]
                coordinates += [keypoints[i], keypoints[i + 1]]
                score += [keypoints[i + 2]]
            skeleton['pose'] = coordinates
            skeleton['score'] = score
            skeletons += [skeleton]
        frame_data['skeleton'] = skeletons
        sequence_info += [frame_data]

    video_info = dict()
    video_info['data'] = sequence_info
    # irrelevant for now..
    video_info['label'] = 'unknown'
    video_info['label_index'] = -1

    return video_info


def prepare_sample(cfg, video_info):
    video_length = len(video_info['data'])
    print('video_length', video_length)
    if cfg.JSON_FORMATTER.T > video_length or cfg.JSON_FORMATTER.T == -1:
        cfg.JSON_FORMATTER.T = video_length
    cfg.JSON_FORMATTER.M = max([len(frame_info['skeleton']) for frame_info in video_info['data']])  # num persons
    data_numpy = np.zeros((cfg.JSON_FORMATTER.C, video_length, cfg.JSON_FORMATTER.V, cfg.JSON_FORMATTER.M))
    for frame_info in video_info['data']:
        frame_index = frame_info['frame_index']
        for m, skeleton_info in enumerate(frame_info["skeleton"]):
            # no need..
            # if m >= cfg.JSON_FORMATTER.M:
            #     break
            pose = skeleton_info['pose']
            score = skeleton_info['score']
            data_numpy[0, frame_index, :, m] = pose[0::2]
            data_numpy[1, frame_index, :, m] = pose[1::2]
            data_numpy[2, frame_index, :, m] = score

    # centralization
    # data_numpy[0:2] = data_numpy[0:2] - 0.5
    # data_numpy[0][data_numpy[2] == 0] = 0
    # data_numpy[1][data_numpy[2] == 0] = 0

    # get & check label index
    label = video_info['label_index']
#     assert (self.label[index] == label)

    # data augmentation
#     if self.random_shift:
#         data_numpy = tools.random_shift(data_numpy)
#     if self.random_choose:
#         data_numpy = tools.random_choose(data_numpy, self.window_size)
#     elif self.window_size > 0:
#         data_numpy = tools.auto_pading(data_numpy, self.window_size)
#     if self.random_move:
#         data_numpy = tools.random_move(data_numpy)

    # sort by score
    sort_index = (-data_numpy[2, :, :, :].sum(axis=1)).argsort(axis=1)
    for t, s in enumerate(sort_index):
        data_numpy[:, t, :, :] = data_numpy[:, t, :, s].transpose((1, 2, 0))
    data_numpy = data_numpy[:, :, :, 0:cfg.JSON_FORMATTER.M]
    # match poses between 2 frames
    if cfg.MATCH_POSE:
        data_numpy = openpose_match(data_numpy)

    data_numpy = rearrange(data_numpy, 'c t k p -> p t k c') # reshape 'coords time kp person -> person time kp coords
    # if cfg.NORM_POSE:
    #     vid_res = list(get_frame_info(cfg.VID_IN))
    #     data_numpy = normalize_pose(data_numpy, vid_res=vid_res) # TODO

    return data_numpy # , label


##############################
# AlphaPose + PoseFlow utils #
##############################



##############################
def apply_transforms(pose_segs_data_np, num_transforms):
    """
    Select sample and augmentation. I.e. given 5 samples and 2 transformations,
    sample 7 is data sample 7%5=2 and transform is 7//5=1
    """
    num_samples = pose_segs_data_np.shape[0]
    ae_trans_list = [
        PoseTransform(sx=1, sy=1, tx=0.0, ty=0, rot=0, flip=False),  # 0
        PoseTransform(sx=1, sy=1, tx=0.0, ty=0, rot=0, flip=True),  # 3
        PoseTransform(sx=1, sy=1, tx=0.0, ty=0, rot=90, flip=False),  # 6
        PoseTransform(sx=1, sy=1, tx=0.0, ty=0, rot=90, flip=True),  # 9
        PoseTransform(sx=1, sy=1, tx=0.0, ty=0, rot=45, flip=False),  # 12
    ]
    trans_list = ae_trans_list[:num_transforms]
    pose_segs_data_np_trans = np.zeros((pose_segs_data_np.shape[0]*num_transforms, *pose_segs_data_np.shape[1:]), dtype=np.float32) # (#person*num_transforms, 18, 12, 3)
    
    for index in range(num_samples*num_transforms):
        sample_index = index % num_samples
        trans_index = index // num_samples
        pose_segs_data_np_trans[index] = trans_list[trans_index](pose_segs_data_np[sample_index])
    return pose_segs_data_np_trans

def get_per_frame_scores(scores, metadata, num_frames=-1):
    """
    aggregates scores to frames and retuers per-frame normality scores
        (was taken for gepc.utils.scoring_utils.get_dataset_scores)
    """
    metadata = np.asarray(metadata)
    person_ids = set(metadata[:,0])
    per_frame_scores = np.repeat(-np.inf, num_frames)
    # per_frame_scores = np.zeros(num_frames)
    clip_person_scores_dict = {i: np.copy(per_frame_scores) for i in person_ids}
    for person_id in person_ids:
        person_metadata_inds = np.where(metadata[:,0] == person_id)[0]
        pid_scores = scores[person_metadata_inds]
        pid_frame_inds = np.array([metadata[i,1] for i in person_metadata_inds])
        clip_person_scores_dict[person_id][pid_frame_inds] = pid_scores
        # pid_scores = scores[person_metadata_inds]
        # pid_frame_inds = np.array([metadata[i][3] for i in person_metadata_inds])
        # clip_person_scores_dict[person_id][pid_frame_inds] = pid_scores

    clip_ppl_score_arr = np.stack(list(clip_person_scores_dict.values()))
    per_frame_scores = np.amax(clip_ppl_score_arr, axis=0)  # per frame 
    per_frame_max_score_pid = [list(person_ids)[i] for i in np.argmax(clip_ppl_score_arr, axis=0)]   # list of per-frame person id with the hightest score
    return per_frame_scores


def align_and_smooth_scores(scores_np, seg_len=12, sigma=40):
    scores_shifted = np.zeros_like(scores_np)
    shift = seg_len + (seg_len // 2) - 1
    scores_shifted[shift:] = scores_np[:-shift]
    scores_smoothed = gaussian_filter1d(scores_shifted, sigma)
    return scores_smoothed


def get_frame_info(video_path, ret_res=True, ret_num_frames=False):
    """
    returns (W,H,T) (depands on flags)
    """
    ret = tuple()
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        if ret_res:
            ret += (cap.get(cv2.CAP_PROP_FRAME_WIDTH),)
            ret += (cap.get(cv2.CAP_PROP_FRAME_HEIGHT),)
        if ret_num_frames:
            ret += (int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),)
        return ret if len(ret) > 1 else ret[0]
    else:
        raise Exception('Could not exstract resolution from {video_path}')


def split_pose_to_segments(single_pose_np, single_pose_keys=None, start_ofst=0, seg_stride=6, seg_len=12, scene_id='', clip_id=''):
    """
    input: np array of shape (P,T,K,3)
    output: (num_seg,K,3), (num_seg,frame_id)
    """
                           
    num_persons, vid_length, kp_count, kp_dim = single_pose_np.shape
    pose_segs_np = np.empty([0, seg_len, kp_count, kp_dim])
    pose_segs_meta = []
    # num_segs = np.ceil((vid_length - seg_len) / seg_stride).astype(np.int)

    for p in range(num_persons):
        frame_idx_appearance = list(np.argwhere(single_pose_np[p,...,-1].sum(-1) > 0).squeeze())
        num_segs = np.ceil((len(frame_idx_appearance) - seg_len) / seg_stride).astype(np.int)
        for seg_ind in range(num_segs):
            start_ind = start_ofst + seg_ind * seg_stride
            start_key = frame_idx_appearance[start_ind]
            if is_seg_continuous(frame_idx_appearance, start_key, seg_len):
                curr_segment = single_pose_np[p,start_ind:start_ind + seg_len].reshape(1, seg_len, kp_count, kp_dim)
                pose_segs_np = np.append(pose_segs_np, curr_segment, axis=0)
                pose_segs_meta.append([p, int(start_key)])

    # single_pose_keys_sorted = sorted([int(i) for i in single_pose_keys])  # , key=lambda x: int(x))
    # for seg_ind in range(num_segs):
    #     start_ind = start_ofst + seg_ind * seg_stride
    #     start_key = single_pose_keys_sorted[start_ind]
    #     if is_seg_continuous(single_pose_keys_sorted, start_key, seg_len):
    #         curr_segment = single_pose_np[start_ind:start_ind + seg_len].reshape(1, seg_len, kp_count, kp_dim)
    #         pose_segs_np = np.append(pose_segs_np, curr_segment, axis=0)
    #         # pose_segs_meta.append([int(scene_id), int(clip_id), int(single_pose_meta[0]), int(start_key)])
    return pose_segs_np, pose_segs_meta


def is_seg_continuous(sorted_seg_keys, start_key, seg_len, missing_th=2):
    """
    Checks if an input clip is continuous or if there are frames missing
    :param sorted_seg_keys:
    :param start_key:
    :param seg_len:
    :param missing_th: The number of frames that are allowed to be missing on a sequence,
    i.e. if missing_th = 1 then a seg for which a single frame is missing is considered continuous
    :return:
    """
    start_idx = sorted_seg_keys.index(start_key)
    expected_idxs = list(range(start_key, start_key + seg_len))
    act_idxs = sorted_seg_keys[start_idx: start_idx + seg_len]
    min_overlap = seg_len - missing_th
    key_overlap = len(set(act_idxs).intersection(expected_idxs))
    if key_overlap >= min_overlap:
        return True
    else:
        return False
