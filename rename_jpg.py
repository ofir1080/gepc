import os
# in_dir = '/datasas/data/ShanghaiTech/training/videos'
work_dir = '/datasas/data/ShanghaiTech/training/frames/01_001'

cwd1 = os.getcwd()
print(cwd1)
os.chdir(work_dir)
cwd2 = os.getcwd()
print(cwd2)

for i in range(1,766):
  # print('mv %03d.jpg %03d.jpg'.format(i, i-1))
  cmd = 'mv' + ' ' + (str(i).zfill(3) + '.jpg') + ' ' + (str(i-1).zfill(3)+'.jpg')
  print(cmd)
  os.system(cmd)

os.chdir(cwd1)
cwd3 = os.getcwd()
print(cwd3)