# split_session.py: 将原始数据集划分为三个session
import os
import shutil
from datetime import datetime

# 定义源目录和目标目录
source_dir = '/home/fb/src/dataset/SEED/ExtractedFeatures'
target_dir = '/home/fb/src/dataset/SEED/ExtractedFeatures'

# 确保目标目录存在
for session in ['session1', 'session2', 'session3']:
    os.makedirs(os.path.join(target_dir, session), exist_ok=True)

# 获取所有被试ID (1-15)
subject_ids = range(1, 16)

# 遍历每个被试
for subject_id in subject_ids:
    # 找出该被试的所有文件
    subject_files = [f for f in os.listdir(source_dir) 
                    if f.startswith(f'{subject_id}_') and f.endswith('.mat')]
    
    # 按日期排序文件
    def get_date_from_filename(filename):
        date_str = filename.split('_')[1].split('.')[0]
        return datetime.strptime(date_str, '%Y%m%d')
    
    subject_files_sorted = sorted(subject_files, key=get_date_from_filename)
    
    for i, filename in enumerate(subject_files_sorted):
        src_path = os.path.join(source_dir, filename)
        session = "session" + str(i % 3 + 1)
        dst_path = os.path.join(target_dir, session, filename)
        shutil.move(src_path, dst_path)
        print(f'Moved {filename} to {session}')

print("所有文件已按被试和session分类完成。")



