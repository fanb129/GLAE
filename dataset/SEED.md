🔍 SEED数据集格式检查
============================================================
📂 数据路径: /home/nise-emo/nise-lab/dataset/public/SEED
🔍 SEED数据集目录结构:
==================================================
SEED/
  seed-stimulation.xlsx
  Channel Order.xlsx
  MyProcessed/
    all_classes/
      readme.txt
    lack_negative/
  Preprocessed_EEG/
    2_20140419.mat
    6_20130712.mat
    4_20140621.mat
    11_20140630.mat
    11_20140625.mat
    ... and 42 more files
  ExtractedFeatures/
    2_20140419.mat
    6_20130712.mat
    4_20140621.mat
    11_20140630.mat
    11_20140625.mat
    ... and 42 more files

🔍 ExtractedFeatures 目录分析:
==================================================
📁 总文件数: 46
👥 被试数量: 15
📅 Session数量: 40
🔢 被试ID: ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']
📅 Session示例: ['20130709', '20130712', '20131016', '20131027', '20131030']...

📄 检查文件: 2_20140419.mat
  📂 加载文件: 2_20140419.mat
  🔑 数据键数量: 180
  📊 数据结构:
    DE特征: de_LDS{trial_idx} (1-15)
    形状: (62, time_windows, 5)
    62: EEG通道数
    time_windows: 时间窗口数（每个trial不同，约185-238个窗口）
    5: 频段数（δ, θ, α, β, γ）
     1. de_movingAve1            : shape=(62, 235, 5)        , dtype=float64
     2. de_LDS1                  : shape=(62, 235, 5)        , dtype=float64
     3. psd_movingAve1           : shape=(62, 235, 5)        , dtype=float64
     4. psd_LDS1                 : shape=(62, 235, 5)        , dtype=float64
     5. dasm_movingAve1          : shape=(27, 235, 5)        , dtype=float64
     6. dasm_LDS1                : shape=(27, 235, 5)        , dtype=float64
     7. rasm_movingAve1          : shape=(27, 235, 5)        , dtype=float64
     8. rasm_LDS1                : shape=(27, 235, 5)        , dtype=float64
     9. asm_movingAve1           : shape=(54, 235, 5)        , dtype=float64
    10. asm_LDS1                 : shape=(54, 235, 5)        , dtype=float64
    11. dcau_movingAve1          : shape=(23, 235, 5)        , dtype=float64
    12. dcau_LDS1                : shape=(23, 235, 5)        , dtype=float64
    13. de_movingAve2            : shape=(62, 233, 5)        , dtype=float64
    14. de_LDS2                  : shape=(62, 233, 5)        , dtype=float64
    15. psd_movingAve2           : shape=(62, 233, 5)        , dtype=float64
    16. psd_LDS2                 : shape=(62, 233, 5)        , dtype=float64
    17. dasm_movingAve2          : shape=(27, 233, 5)        , dtype=float64
    18. dasm_LDS2                : shape=(27, 233, 5)        , dtype=float64
    19. rasm_movingAve2          : shape=(27, 233, 5)        , dtype=float64
    20. rasm_LDS2                : shape=(27, 233, 5)        , dtype=float64
    21. asm_movingAve2           : shape=(54, 233, 5)        , dtype=float64
    22. asm_LDS2                 : shape=(54, 233, 5)        , dtype=float64
    23. dcau_movingAve2          : shape=(23, 233, 5)        , dtype=float64
    24. dcau_LDS2                : shape=(23, 233, 5)        , dtype=float64
    25. de_movingAve3            : shape=(62, 206, 5)        , dtype=float64
    26. de_LDS3                  : shape=(62, 206, 5)        , dtype=float64
    27. psd_movingAve3           : shape=(62, 206, 5)        , dtype=float64
    28. psd_LDS3                 : shape=(62, 206, 5)        , dtype=float64
    29. dasm_movingAve3          : shape=(27, 206, 5)        , dtype=float64
    30. dasm_LDS3                : shape=(27, 206, 5)        , dtype=float64
    31. rasm_movingAve3          : shape=(27, 206, 5)        , dtype=float64
    32. rasm_LDS3                : shape=(27, 206, 5)        , dtype=float64
    33. asm_movingAve3           : shape=(54, 206, 5)        , dtype=float64
    34. asm_LDS3                 : shape=(54, 206, 5)        , dtype=float64
    35. dcau_movingAve3          : shape=(23, 206, 5)        , dtype=float64
    36. dcau_LDS3                : shape=(23, 206, 5)        , dtype=float64
    37. de_movingAve4            : shape=(62, 238, 5)        , dtype=float64
    38. de_LDS4                  : shape=(62, 238, 5)        , dtype=float64
    39. psd_movingAve4           : shape=(62, 238, 5)        , dtype=float64
    40. psd_LDS4                 : shape=(62, 238, 5)        , dtype=float64
    41. dasm_movingAve4          : shape=(27, 238, 5)        , dtype=float64
    42. dasm_LDS4                : shape=(27, 238, 5)        , dtype=float64
    43. rasm_movingAve4          : shape=(27, 238, 5)        , dtype=float64
    44. rasm_LDS4                : shape=(27, 238, 5)        , dtype=float64
    45. asm_movingAve4           : shape=(54, 238, 5)        , dtype=float64
    46. asm_LDS4                 : shape=(54, 238, 5)        , dtype=float64
    47. dcau_movingAve4          : shape=(23, 238, 5)        , dtype=float64
    48. dcau_LDS4                : shape=(23, 238, 5)        , dtype=float64
    49. de_movingAve5            : shape=(62, 185, 5)        , dtype=float64
    50. de_LDS5                  : shape=(62, 185, 5)        , dtype=float64
    51. psd_movingAve5           : shape=(62, 185, 5)        , dtype=float64
    52. psd_LDS5                 : shape=(62, 185, 5)        , dtype=float64
    53. dasm_movingAve5          : shape=(27, 185, 5)        , dtype=float64
    54. dasm_LDS5                : shape=(27, 185, 5)        , dtype=float64
    55. rasm_movingAve5          : shape=(27, 185, 5)        , dtype=float64
    56. rasm_LDS5                : shape=(27, 185, 5)        , dtype=float64
    57. asm_movingAve5           : shape=(54, 185, 5)        , dtype=float64
    58. asm_LDS5                 : shape=(54, 185, 5)        , dtype=float64
    59. dcau_movingAve5          : shape=(23, 185, 5)        , dtype=float64
    60. dcau_LDS5                : shape=(23, 185, 5)        , dtype=float64
    61. de_movingAve6            : shape=(62, 195, 5)        , dtype=float64
    62. de_LDS6                  : shape=(62, 195, 5)        , dtype=float64
    63. psd_movingAve6           : shape=(62, 195, 5)        , dtype=float64
    64. psd_LDS6                 : shape=(62, 195, 5)        , dtype=float64
    65. dasm_movingAve6          : shape=(27, 195, 5)        , dtype=float64
    66. dasm_LDS6                : shape=(27, 195, 5)        , dtype=float64
    67. rasm_movingAve6          : shape=(27, 195, 5)        , dtype=float64
    68. rasm_LDS6                : shape=(27, 195, 5)        , dtype=float64
    69. asm_movingAve6           : shape=(54, 195, 5)        , dtype=float64
    70. asm_LDS6                 : shape=(54, 195, 5)        , dtype=float64
    71. dcau_movingAve6          : shape=(23, 195, 5)        , dtype=float64
    72. dcau_LDS6                : shape=(23, 195, 5)        , dtype=float64
    73. de_movingAve7            : shape=(62, 237, 5)        , dtype=float64
    74. de_LDS7                  : shape=(62, 237, 5)        , dtype=float64
    75. psd_movingAve7           : shape=(62, 237, 5)        , dtype=float64
    76. psd_LDS7                 : shape=(62, 237, 5)        , dtype=float64
    77. dasm_movingAve7          : shape=(27, 237, 5)        , dtype=float64
    78. dasm_LDS7                : shape=(27, 237, 5)        , dtype=float64
    79. rasm_movingAve7          : shape=(27, 237, 5)        , dtype=float64
    80. rasm_LDS7                : shape=(27, 237, 5)        , dtype=float64
    81. asm_movingAve7           : shape=(54, 237, 5)        , dtype=float64
    82. asm_LDS7                 : shape=(54, 237, 5)        , dtype=float64
    83. dcau_movingAve7          : shape=(23, 237, 5)        , dtype=float64
    84. dcau_LDS7                : shape=(23, 237, 5)        , dtype=float64
    85. de_movingAve8            : shape=(62, 216, 5)        , dtype=float64
    86. de_LDS8                  : shape=(62, 216, 5)        , dtype=float64
    87. psd_movingAve8           : shape=(62, 216, 5)        , dtype=float64
    88. psd_LDS8                 : shape=(62, 216, 5)        , dtype=float64
    89. dasm_movingAve8          : shape=(27, 216, 5)        , dtype=float64
    90. dasm_LDS8                : shape=(27, 216, 5)        , dtype=float64
    91. rasm_movingAve8          : shape=(27, 216, 5)        , dtype=float64
    92. rasm_LDS8                : shape=(27, 216, 5)        , dtype=float64
    93. asm_movingAve8           : shape=(54, 216, 5)        , dtype=float64
    94. asm_LDS8                 : shape=(54, 216, 5)        , dtype=float64
    95. dcau_movingAve8          : shape=(23, 216, 5)        , dtype=float64
    96. dcau_LDS8                : shape=(23, 216, 5)        , dtype=float64
    97. de_movingAve9            : shape=(62, 265, 5)        , dtype=float64
    98. de_LDS9                  : shape=(62, 265, 5)        , dtype=float64
    99. psd_movingAve9           : shape=(62, 265, 5)        , dtype=float64
    100. psd_LDS9                 : shape=(62, 265, 5)        , dtype=float64
    101. dasm_movingAve9          : shape=(27, 265, 5)        , dtype=float64
    102. dasm_LDS9                : shape=(27, 265, 5)        , dtype=float64
    103. rasm_movingAve9          : shape=(27, 265, 5)        , dtype=float64
    104. rasm_LDS9                : shape=(27, 265, 5)        , dtype=float64
    105. asm_movingAve9           : shape=(54, 265, 5)        , dtype=float64
    106. asm_LDS9                 : shape=(54, 265, 5)        , dtype=float64
    107. dcau_movingAve9          : shape=(23, 265, 5)        , dtype=float64
    108. dcau_LDS9                : shape=(23, 265, 5)        , dtype=float64
    109. de_movingAve10           : shape=(62, 237, 5)        , dtype=float64
    110. de_LDS10                 : shape=(62, 237, 5)        , dtype=float64
    111. psd_movingAve10          : shape=(62, 237, 5)        , dtype=float64
    112. psd_LDS10                : shape=(62, 237, 5)        , dtype=float64
    113. dasm_movingAve10         : shape=(27, 237, 5)        , dtype=float64
    114. dasm_LDS10               : shape=(27, 237, 5)        , dtype=float64
    115. rasm_movingAve10         : shape=(27, 237, 5)        , dtype=float64
    116. rasm_LDS10               : shape=(27, 237, 5)        , dtype=float64
    117. asm_movingAve10          : shape=(54, 237, 5)        , dtype=float64
    118. asm_LDS10                : shape=(54, 237, 5)        , dtype=float64
    119. dcau_movingAve10         : shape=(23, 237, 5)        , dtype=float64
    120. dcau_LDS10               : shape=(23, 237, 5)        , dtype=float64
    121. de_movingAve11           : shape=(62, 235, 5)        , dtype=float64
    122. de_LDS11                 : shape=(62, 235, 5)        , dtype=float64
    123. psd_movingAve11          : shape=(62, 235, 5)        , dtype=float64
    124. psd_LDS11                : shape=(62, 235, 5)        , dtype=float64
    125. dasm_movingAve11         : shape=(27, 235, 5)        , dtype=float64
    126. dasm_LDS11               : shape=(27, 235, 5)        , dtype=float64
    127. rasm_movingAve11         : shape=(27, 235, 5)        , dtype=float64
    128. rasm_LDS11               : shape=(27, 235, 5)        , dtype=float64
    129. asm_movingAve11          : shape=(54, 235, 5)        , dtype=float64
    130. asm_LDS11                : shape=(54, 235, 5)        , dtype=float64
    131. dcau_movingAve11         : shape=(23, 235, 5)        , dtype=float64
    132. dcau_LDS11               : shape=(23, 235, 5)        , dtype=float64
    133. de_movingAve12           : shape=(62, 233, 5)        , dtype=float64
    134. de_LDS12                 : shape=(62, 233, 5)        , dtype=float64
    135. psd_movingAve12          : shape=(62, 233, 5)        , dtype=float64
    136. psd_LDS12                : shape=(62, 233, 5)        , dtype=float64
    137. dasm_movingAve12         : shape=(27, 233, 5)        , dtype=float64
    138. dasm_LDS12               : shape=(27, 233, 5)        , dtype=float64
    139. rasm_movingAve12         : shape=(27, 233, 5)        , dtype=float64
    140. rasm_LDS12               : shape=(27, 233, 5)        , dtype=float64
    141. asm_movingAve12          : shape=(54, 233, 5)        , dtype=float64
    142. asm_LDS12                : shape=(54, 233, 5)        , dtype=float64
    143. dcau_movingAve12         : shape=(23, 233, 5)        , dtype=float64
    144. dcau_LDS12               : shape=(23, 233, 5)        , dtype=float64
    145. de_movingAve13           : shape=(62, 235, 5)        , dtype=float64
    146. de_LDS13                 : shape=(62, 235, 5)        , dtype=float64
    147. psd_movingAve13          : shape=(62, 235, 5)        , dtype=float64
    148. psd_LDS13                : shape=(62, 235, 5)        , dtype=float64
    149. dasm_movingAve13         : shape=(27, 235, 5)        , dtype=float64
    150. dasm_LDS13               : shape=(27, 235, 5)        , dtype=float64
    151. rasm_movingAve13         : shape=(27, 235, 5)        , dtype=float64
    152. rasm_LDS13               : shape=(27, 235, 5)        , dtype=float64
    153. asm_movingAve13          : shape=(54, 235, 5)        , dtype=float64
    154. asm_LDS13                : shape=(54, 235, 5)        , dtype=float64
    155. dcau_movingAve13         : shape=(23, 235, 5)        , dtype=float64
    156. dcau_LDS13               : shape=(23, 235, 5)        , dtype=float64
    157. de_movingAve14           : shape=(62, 238, 5)        , dtype=float64
    158. de_LDS14                 : shape=(62, 238, 5)        , dtype=float64
    159. psd_movingAve14          : shape=(62, 238, 5)        , dtype=float64
    160. psd_LDS14                : shape=(62, 238, 5)        , dtype=float64
    161. dasm_movingAve14         : shape=(27, 238, 5)        , dtype=float64
    162. dasm_LDS14               : shape=(27, 238, 5)        , dtype=float64
    163. rasm_movingAve14         : shape=(27, 238, 5)        , dtype=float64
    164. rasm_LDS14               : shape=(27, 238, 5)        , dtype=float64
    165. asm_movingAve14          : shape=(54, 238, 5)        , dtype=float64
    166. asm_LDS14                : shape=(54, 238, 5)        , dtype=float64
    167. dcau_movingAve14         : shape=(23, 238, 5)        , dtype=float64
    168. dcau_LDS14               : shape=(23, 238, 5)        , dtype=float64
    169. de_movingAve15           : shape=(62, 206, 5)        , dtype=float64
    170. de_LDS15                 : shape=(62, 206, 5)        , dtype=float64
    171. psd_movingAve15          : shape=(62, 206, 5)        , dtype=float64
    172. psd_LDS15                : shape=(62, 206, 5)        , dtype=float64
    173. dasm_movingAve15         : shape=(27, 206, 5)        , dtype=float64
    174. dasm_LDS15               : shape=(27, 206, 5)        , dtype=float64
    175. rasm_movingAve15         : shape=(27, 206, 5)        , dtype=float64
    176. rasm_LDS15               : shape=(27, 206, 5)        , dtype=float64
    177. asm_movingAve15          : shape=(54, 206, 5)        , dtype=float64
    178. asm_LDS15                : shape=(54, 206, 5)        , dtype=float64
    179. dcau_movingAve15         : shape=(23, 206, 5)        , dtype=float64
    180. dcau_LDS15               : shape=(23, 206, 5)        , dtype=float64

  🧠 ExtractedFeatures 详细分析:
    🎯 找到 30 个DE特征键
    📝 Trial DE特征: 15 个
      Trial 1 (de_LDS1): (62, 235, 5)
        - 通道数: 62
        - 时间窗口: 235
        - 频段数: 5
        - 数值范围: [10.849, 27.315]
      Trial 2 (de_LDS2): (62, 233, 5)
        - 通道数: 62
        - 时间窗口: 233
        - 频段数: 5
        - 数值范围: [10.846, 27.003]
      Trial 3 (de_LDS3): (62, 206, 5)
        - 通道数: 62
        - 时间窗口: 206
        - 频段数: 5
        - 数值范围: [10.790, 28.381]
    🏷️  特征类型: ['asm', 'dasm', 'dcau', 'de', 'psd', 'rasm']

🔍 Preprocessed_EEG 目录分析:
==================================================
📁 总文件数: 46

📄 检查文件: 2_20140419.mat
  📂 加载文件: 2_20140419.mat
  🔑 数据键数量: 15
  📊 数据结构:
     1. jl_eeg1                  : shape=(62, 47001)         , dtype=float64
     2. jl_eeg2                  : shape=(62, 46601)         , dtype=float64
     3. jl_eeg3                  : shape=(62, 41201)         , dtype=float64
     4. jl_eeg4                  : shape=(62, 47601)         , dtype=float64
     5. jl_eeg5                  : shape=(62, 37001)         , dtype=float64
     6. jl_eeg6                  : shape=(62, 39001)         , dtype=float64
     7. jl_eeg7                  : shape=(62, 47401)         , dtype=float64
     8. jl_eeg8                  : shape=(62, 43201)         , dtype=float64
     9. jl_eeg9                  : shape=(62, 53001)         , dtype=float64
    10. jl_eeg10                 : shape=(62, 47401)         , dtype=float64
    11. jl_eeg11                 : shape=(62, 47001)         , dtype=float64
    12. jl_eeg12                 : shape=(62, 46601)         , dtype=float64
    13. jl_eeg13                 : shape=(62, 47001)         , dtype=float64
    14. jl_eeg14                 : shape=(62, 47601)         , dtype=float64
    15. jl_eeg15                 : shape=(62, 41201)         , dtype=float64

  🧠 Preprocessed_EEG 详细分析:
    🎯 找到 15 个EEG数据键
      Trial 1 (jl_eeg1): (62, 47001)
        - 通道数: 62
        - 时间点数: 47001
        - 采样时长: 235.0秒 (假设200Hz)
        - 数值范围: [-187.308, 147.879]
      Trial 2 (jl_eeg2): (62, 46601)
        - 通道数: 62
        - 时间点数: 46601
        - 采样时长: 233.0秒 (假设200Hz)
        - 数值范围: [-146.925, 204.355]
      Trial 3 (jl_eeg3): (62, 41201)
        - 通道数: 62
        - 时间点数: 41201
        - 采样时长: 206.0秒 (假设200Hz)
        - 数值范围: [-1305.848, 1089.871]

🏷️  SEED数据集标签信息:
==================================================
📋 标准标签映射:
  正面情绪 (Positive): 0
  中性情绪 (Neutral):  1
  负面情绪 (Negative): 2

📝 Trial到情绪的映射:
  正面情绪 trials: [1, 2, 3, 13, 14, 15]
  中性情绪 trials: [4, 5, 6, 10, 11, 12]
  负面情绪 trials: [7, 8, 9]

🎬 电影片段信息:
  每个trial对应一个情感电影片段
  每个片段约4分钟
  总共15个片段，每类情绪5个片段

📊 刺激文件分析:
==================================================
  ✅ 找到刺激文件: seed-stimulation.xlsx
  📏 数据形状: (17, 5)
  🔑 列名: ['Name of the clip', 'Label', 'Source url', 'Start time point', 'End time point']

  📋 前5行数据:
       Name of the clip  Label                                                                                                        Source url Start time point End time point
       Lost in Thailand    2.0                                                                     https://v.qq.com/x/cover/tjtiyg1qe0zkhdk.html         00:06:13       00:10:11
World Heritage in China    1.0                                                                          https://v.qq.com/x/page/c0156s8g85t.html         00:00:50       00:04:36
             Aftershock    0.0 https://v.youku.com/v_show/id_XNDcxNTg4OTk2.html?spm=a2h0k.11417342.soresults.dplaybutton&lang=%E5%9B%BD%E8%AF%AD         00:20:10       00:23:35
           Back to 1942    0.0                                                         https://v.qq.com/x/cover/phk4s9bkfs8xdn4/r0012fmwaf2.html         00:49:58       00:54:00
World Heritage in China    1.0                                                                          https://v.qq.com/x/page/c0156s8g85t.html         00:10:40       00:13:44

✅ 数据格式检查完成!