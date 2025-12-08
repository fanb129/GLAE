import scipy.io as scio
 
label_file = '/home/nise-emo/nise-lab/dataset/public/SEED/ExtractedFeatures/label.mat'
label_data = scio.loadmat(label_file)
print(label_data['label'].shape)
print(label_data['label'])

data_file = '/home/nise-emo/nise-lab/dataset/public/SEED/ExtractedFeatures/10_20131130.mat'
data = scio.loadmat(data_file)
for key in data.keys():
    if key.startswith('de_LDS'):
        print(key, data[key].shape)

######SEED_IV####
#The labels with 0, 1, 2, and 3 denote the ground truth, neutral, sad, fear, and happy emotions, respectively.
session1_label = [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3]
session2_label = [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1]
session3_label = [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]
print(len(session1_label))

data_file_seed_iv = '/home/nise-emo/nise-lab/dataset/public/SEED_IV/eeg_feature_smooth/session1/10_20151014.mat'
data_seed_iv = scio.loadmat(data_file_seed_iv)
for key in data_seed_iv.keys():
    if key.startswith('de_LDS'):
        print(key, data_seed_iv[key].shape)

'''
(1, 15)
[[ 1  0 -1 -1  0  1 -1  0  1  1  0 -1  0  1 -1]]
de_LDS1 (62, 235, 5)
de_LDS2 (62, 233, 5)
de_LDS3 (62, 206, 5)
de_LDS4 (62, 238, 5)
de_LDS5 (62, 185, 5)
de_LDS6 (62, 195, 5)
de_LDS7 (62, 237, 5)
de_LDS8 (62, 216, 5)
de_LDS9 (62, 265, 5)
de_LDS10 (62, 237, 5)
de_LDS11 (62, 235, 5)
de_LDS12 (62, 233, 5)
de_LDS13 (62, 235, 5)
de_LDS14 (62, 238, 5)
de_LDS15 (62, 206, 5)
24
de_LDS1 (62, 42, 5)
de_LDS2 (62, 23, 5)
de_LDS3 (62, 49, 5)
de_LDS4 (62, 32, 5)
de_LDS5 (62, 22, 5)
de_LDS6 (62, 40, 5)
de_LDS7 (62, 38, 5)
de_LDS8 (62, 52, 5)
de_LDS9 (62, 36, 5)
de_LDS10 (62, 42, 5)
de_LDS11 (62, 12, 5)
de_LDS12 (62, 27, 5)
de_LDS13 (62, 54, 5)
de_LDS14 (62, 42, 5)
de_LDS15 (62, 64, 5)
de_LDS16 (62, 35, 5)
de_LDS17 (62, 17, 5)
de_LDS18 (62, 44, 5)
de_LDS19 (62, 35, 5)
de_LDS20 (62, 12, 5)
de_LDS21 (62, 28, 5)
de_LDS22 (62, 28, 5)
de_LDS23 (62, 43, 5)
de_LDS24 (62, 34, 5)

==== Dataset Info ====
Source subject 0: 2959 samples
Source subject 1: 2959 samples
Source subject 2: 2959 samples
Source subject 3: 2959 samples
Source subject 4: 2959 samples
Source subject 5: 2959 samples
Source subject 6: 2959 samples
Source subject 7: 2959 samples
Source subject 8: 2959 samples
Source subject 9: 2959 samples
Source subject 10: 2959 samples
Source subject 11: 2959 samples
Source subject 12: 2959 samples
Source subject 13: 2959 samples
41426
Target subject: 2959 samples
44385

==== Dataset Info ====
Source subject 0: 635 samples
Source subject 1: 635 samples
Source subject 2: 635 samples
Source subject 3: 635 samples
Source subject 4: 635 samples
Source subject 5: 635 samples
Source subject 6: 635 samples
Source subject 7: 635 samples
Source subject 8: 635 samples
Source subject 9: 635 samples
Source subject 10: 635 samples
Source subject 11: 635 samples
Source subject 12: 635 samples
Source subject 13: 635 samples
8890
Target subject: 635 samples
9525
'''