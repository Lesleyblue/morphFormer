#---------------------------------------------------#
#   /home/lxl/project/semi-lp/pseudo_loss_k_2d.py  删掉聚类方法的——监督学习
#   网络用transformer
#   loss = 标记标签的loss+伪标签的loss
#---------------------------------------------------#
import numpy as np
import torch
""" 1. 导入数据并显示原图像、标准化数据"""
from matplotlib import pyplot as plt
from scipy.io import loadmat
import spectral as spy
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
from utils import*
import torch.optim as optim
import torch.utils.data as data
from torchsummary import summary
from tqdm import tqdm
import time
from datetime import datetime
import random
import pandas as pd
from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.metrics import classification_report,confusion_matrix
import torch.utils.data as Data
import torch.nn as nn
from scipy.stats import mode
from model import CNN
#---------------------------------------------------#
#   设置种子
#---------------------------------------------------#

best_result = 0
accuary_result = np.empty(shape=(0,), dtype = np.int32)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 读取数据
img, gt, label_values, datasets_name = loadIndianPinesData()
class_num = len(np.unique(gt.reshape(gt.shape[0]*gt.shape[1],1)))-1
"""参数设置"""
patch_size = 15
train_size=20
batch_size=64
epoch=200
lambdas = 0.03
FM = 16
net = CNN(FM,img.shape[2],class_num,True).to(device)
LR = 5e-4
optimizer = torch.optim.Adam(net.parameters(), lr=LR,weight_decay=5e-3)
numComponents = 64
nan_mask = np.isnan(img.sum(axis=-1))  # img.sum(axis=-1) 将最后一轴求和，这里是将光谱通道上的值都加在一起
if np.count_nonzero(nan_mask) > 0:   # 首先检查 nan_mask 数组中非零元素的数量是否大于0，即是否存在NaN值。
    print(
        "Warning: NaN have been found in the data. It is preferable to remove them beforehand. Learning on NaN data is disabled."
    )
img[nan_mask] = 0  # 将img数组中在nan_mask对应位置为True的元素（即求和结果为NaN的像素）设置为0。这样做的目的是将NaN值的像素替换为0
gt[nan_mask] = 0
"""标准化数据"""
X = img.astype(np.float32)
X = data_preprocessing(X) 
XPatches, yPatches = createPatches(X, gt, windowSize=patch_size)
yPatches = yPatches.astype(np.int32)
""" 2. 将gt转为画板上的颜色、显示类别对应的颜色"""
palette = {0: (0, 0, 0)}
for k, color in enumerate(sns.color_palette('hls', class_num)):   # 使用"hls"调色板生成16个颜色
    palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype="uint8"))

"""3. 标记数据分为训练集和测试集, 未标数据集"""
test_percalss_num, train_gt,test_gt = sample_gt_2(gt, train_size)
parameter = torch.load("PC_params/MF_20_97.67.pth", map_location=lambda storage, loc: storage.cuda())
net.load_state_dict(parameter)
"""6. 预测"""
net.eval()
# probabilities = np.zeros(img.shape[:2] + (len(label_values),))  # probabilities = (610, 340, 10)
probabilities = np.zeros(img.shape[:2] + (64,))   # probabilities = (512, 217)
pred = np.zeros((img.shape[0]*img.shape[1], 64), dtype=np.float32) # (111104, 64)
start_eval_time = time.time()
with torch.no_grad():
    datas = XPatches # 取图像块   
    datas = datas.transpose((0, 3, 1, 2))
    datas = torch.FloatTensor(datas)
    # 创建数据集
    dataset = Data.TensorDataset(datas)
    # 创建数据加载器，注意预测时不需要打乱数据
    data_loader = data.DataLoader(dataset, batch_size=64, shuffle=False, drop_last=False)
    # 存储预测结果
    predictions = []
    # 确保网络处于评估模式
    net.eval()
    for idx,batch in enumerate(data_loader):
        # 将数据移至GPU         
        batch = batch[0].to(device)
        # 通过网络进行预测
        output= net(batch)    # torch.Size([64, 64])
        # 存储预测结果
        pred[idx*output.shape[0]:idx*output.shape[0]+output.shape[0],:] = output.cpu()
    i=0
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            probabilities[x,y]+=pred[i]
            i+=1
end_eval_time = time.time()
eval_time = end_eval_time - start_eval_time
print("eval time is %s s" % eval_time)
prediction = np.argmax(probabilities, axis=-1)  # 沿着最后一个维度（axis=-1）进行求最大值的索引操作
print("test_gt", np.max(test_gt))
prediction = prediction + 1
run_results = metrics(
    prediction,
    test_gt,
    test_percalss_num,   # 每类的测试样本
    None
)
mask = np.zeros(gt.shape, dtype="bool")
mask[gt == 0] = True
if datasets_name!='KSC':
    prediction[mask] = 0
oa, text = show_results(run_results, label_values)
# 可视化
pre = convert_to_color(prediction, palette)
dpi = 300
# save_path = 'clear_images/PU_predition_train_size=20_{:.2f}.png'.format(oa)
save_path = 'clear_images/PC_predition_train_size=20_{:.2f}.png'.format(oa)
# save_path = 'clear_images/PU_predition_train_size=10_{}.png'.format(oa)
fig = plt.figure(frameon=False)
fig.set_size_inches(gt.shape[1]*2.0/dpi, gt.shape[0]*2.0/dpi)

ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
fig.add_axes(ax)

ax.imshow(pre)
fig.savefig(save_path, dpi=dpi)
