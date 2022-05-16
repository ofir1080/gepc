# Set data root directory
# data_dir = '~/gepc/data'

# Normal (0) / Abnormal (1) binary per-frame .npy Ground Truth files

# Remove all ground truth files from test_frame_mask directory
# rm ~/gepc/data/testing/test_frame_mask/*.npy

# Copy ground truth files to test_frame_mask directory according to requested number of test files
# Single file
# cp ~/gepc/data/testing/test_frame_mask_orig/01_0014.npy ~/gepc/data/testing/test_frame_mask
# All files
# cp ~/gepc/data/testing/test_frame_mask_orig/*.npy ~/gepc/data/testing/test_frame_mask

# Tracked person .json files

# Remove all tracking json test files from tracked_person directory
# rm ~/gepc/data/pose/testing/tracked_person/*.json
# rm ~/gepc/data/pose/training/tracked_person/*.json

# Copy tracking json test files tracked_person_[amir|joni|posetrack] to tracked_person
# Single file
# cp ~/gepc/data/pose/testing/tracked_person_amir/01_0014_alphapose_tracked_person.json ~/gepc/data/pose/testing/tracked_person
# cp ~/gepc/data/pose/testing/tracked_person_joni/01_0014_alphapose_tracked_person.json ~/gepc/data/pose/testing/tracked_person
# cp ~/gepc/data/pose/testing/tracked_person_posetrack/01_0014_alphapose_tracked_person.json ~/gepc/data/pose/testing/tracked_person
# All files
# cp ~/gepc/data/pose/testing/tracked_person_amir/*.json ~/gepc/data/pose/testing/tracked_person
# cp ~/gepc/data/pose/testing/tracked_person_joni/*.json ~/gepc/data/pose/testing/tracked_person
# cp ~/gepc/data/pose/testing/tracked_person_posetrack/*.json ~/gepc/data/pose/testing/tracked_person
# cp ~/gepc/data/pose/training/tracked_person_posetrack/*.json ~/gepc/data/pose/training/tracked_person

# Copy from AlphaPose results dir to gepc results dir
# find . -name "*alphapose_tracked_person.json" -exec cp {} ~/gepc/data/pose/testing/tracked_person_joni \;
# find . -name "*alphapose_tracked_person.json" -type f -exec cp {} ~/gepc/data/pose/testing/tracked_person_joni

# (base) xpct84@zil56lxapp07:~/AlphaPose/examples/res/ShanghaiTech/testing_pose_track$ find . -name "*alphapose_tracked_person.json" -exec cp {} ~/gepc/data/pose/testing/tracked_person_posetrack \;
# (base) xpct84@zil56lxapp07:~/AlphaPose/examples/res/ShanghaiTech/training_pose_track$ find . -name "*alphapose_tracked_person.json" -exec cp {} ~/gepc/data/pose/training/tracked_person_posetrack \;