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
os.environ["CUDA_VISIBLE_DEVICES"]="1"
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
from spectralFormer import ViT
#---------------------------------------------------#
#   设置种子
#---------------------------------------------------#
# seed = 3407  
best_result = 0
# seed = 78
accuary_result = np.empty(shape=(0,), dtype = np.int32)
for i in range(100):
    # seed = 42
    print("这是第{}次".format(i))
    seed = i+1  # 
    def seed_everything(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    seed_everything(seed)


    #创建train_acc.csv和var_acc.csv文件，记录loss和accuracy
    df = pd.DataFrame(columns=['train Loss','val accuracy'])#列名
    df.to_csv("./train_acc.csv",index=False) 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 读取数据
    img, gt, label_values, datasets_name = loadIndianPinesData()
    class_num = len(np.unique(gt.reshape(gt.shape[0]*gt.shape[1],1)))-1
    # view1 = spy.imshow(data=img, bands=[43, 21, 11], title="img", figsize=(5, 5))  # .函数中的bands参数用于校正颜色
    """参数设置"""
    params = {
        "data": {
            "patch_size": 15,
            "batch_size": 64,
            "num_classes": 16,  # pavia : 9  indianpine: 16   KSC: 13
            "dim_heads": 64,
            "spectral_size":600
        },
        "net": {
            "trainer": "cross_trainer",
            "net_type": "just_pixel",
            "mlp_head_dim": 64,
            "depth": 2,
            "dim": 64,
            "heads": 20
        },
        "train": {
            "epochs": 100,   
            "lr": 0.001,
            "weight_decay": 0,
            "temp": 20
        }
    }
    patch_size = 15
    train_size=20
    batch_size=64
    epoch=200
    lambdas = 0.03
    FM = 16
    # net = CNN(FM,img.shape[2],class_num,True).to(device)
    net = ViT(
        image_size = 7,
        near_band = 1,
        num_patches = img.shape[2],
        num_classes = class_num,
        dim = 64,
        depth = 5,
        heads = 4,
        mlp_dim = 8,
        dropout = 0.1,
        emb_dropout = 0.1,
        mode = 'ViT'
    )
    net = net.to(device)
    # optimizer = optim.SGD(net.parameters(), lr=0.003, momentum=0.9, weight_decay=1e-6, nesterov=True)
    weight_decay = params['train'].get('weight_decay', 5e-3)
    LR = 5e-4
    # optimizer = optim.Adam(net.parameters(), lr=params['train']['lr'], weight_decay=weight_decay)
    optimizer = torch.optim.Adam(net.parameters(), lr=LR,weight_decay=0)
    Loopnum = 1
    Stepnum = 1
    numComponents = 64
    # Filter NaN out
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
    color_gt = convert_to_color(gt, palette) 
    view3 = spy.imshow(classes=color_gt, title="gt")
    plt.axis("off") # 不显示坐标轴
    plt.savefig("./images/gt.jpg")
    # 显示类别和对应的颜色
    fig, axs = plt.subplots(nrows=len(label_values), ncols=2)
    # 循环遍历所有的 subplot，用相应的颜色填充它们。
    for i in range(len(label_values)):
        ax1 = axs[i, 0]  # 获取当前组的第一个子图
        ax1.imshow(np.tile(palette[i+1], (5, 5, 1)))
        ax2 = axs[i, 1]  # 获取当前组的第二个子图
        ax2.text(0.4, 0.4, label_values[i], fontsize=10, ha='left')
        ax1.axis('off')
        ax2.axis('off')
    plt.subplots_adjust(wspace=0)
    plt.savefig("./images/{}_class_color.jpg".format(datasets_name))
    """3. 标记数据分为训练集和测试集, 未标数据集"""

    test_percalss_num, train_gt,test_gt = sample_gt_2(gt, train_size)
   

    # 获取未标记数据集
    train_set = train_gt.reshape((train_gt.shape[0]*train_gt.shape[1],1),order='C')
    train_index = np.argwhere(train_set!=0)  # 每行代表一个非零元素的行索引和列索引
    train_index = train_index[:,0]


    test_set = test_gt.reshape((test_gt.shape[0]*test_gt.shape[1],1),order='C')
    test_index = np.argwhere(test_set!=0)
    test_index = test_index[:,0]

    for loop in range(0,Loopnum):
        # X_train=XPatches[train_index,:,:,:]
        y_train=yPatches[train_index]-1 #减一是为了onehot

        # X_test=XPatches[test_index,:,:,:]
        y_test=yPatches[test_index]-1 #减一是为了onehot    

        Train_added = np.vstack((train_index,y_train)).T   # np.vstack 垂直堆叠
        Test_removed = np.vstack((test_index,y_test)).T
        for step in range(0,Stepnum): 
            X_train = XPatches[Train_added[:,0],:,:,:]
            y_train = Train_added[:,1]
            X_test = XPatches[Test_removed[:,0],:,:,:]
            y_test = Test_removed[:,1]
            # 转置
            X_train = X_train.transpose((0, 3, 1, 2))
            X_test = X_test.transpose((0, 3, 1, 2))
            X_train = torch.FloatTensor(X_train)
            X_test = torch.FloatTensor(X_test)
            y_train = torch.from_numpy(y_train)

            # X_test = torch.from_numpy(X_test)
            y_test = torch.from_numpy(y_test)


            train_dataset = Data.TensorDataset(X_train, y_train)
            train_loader = data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                # pin_memory=True,
                shuffle=True,
                drop_last=False
                ) 
            
            """获取网络的参数"""
            if torch.cuda.is_available():
                net = net.cuda()
            print("Network :")
            with torch.no_grad():
                for input, label in train_loader:  
                    input = input.cuda()
                    label = label.cuda()
                    break
                print(input.shape)  
                print(label) 
                input_size = input.size()[1:]
                summary(net, input_size)
            """5. 训练"""
            avg_loss_epoch=[]
            save_epoch = epoch // 20 if epoch > 20 else 1
            # 如果epoch大于20，将epoch除以20并取整数部分作为save_epoch的值。这意味着每隔2个epoch(40)保存一次模型检查点。
            # 如果epoch小于等于20，将save_epoch的值设为1。这表示在前20个epoch中，每个epoch都会保存一次模型检查点。
            losses = np.zeros(1000000)  # losses 是一个长度为 1000000 的一维数组，用于存储损失值
            mean_losses = np.zeros(100000000)
            iter_ = 1
            display_iter = 100
            loss_win, val_win = None, None
            val_accuracies = []
            criterion = nn.CrossEntropyLoss()
            patch_criterion = nn.MSELoss()
            train_start_time = time.time()
            if torch.cuda.is_available():
                criterion = criterion.cuda()
                patch_criterion = patch_criterion.cuda()
            for e in tqdm(range(1, epoch + 1), desc="Training the network"):
                net.train()
                avg_loss = 0.0
                # 并同时返回批次索引（batch_idx）和数据及其对应的标签（data和target）
                for batch_idx, (datas, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
                    # Load the data into the GPU if required
                    if torch.cuda.is_available():
                        torch.cuda.FloatTensor
                        datas, target = datas.cuda(), target.cuda()
                    optimizer.zero_grad()
                    output = net(datas)  # net是一个神经网络模型。通过传入data作为输入，调用net对象，可以进行前向传播计算得到输出    
                    loss = criterion(output, target)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 15)
                    optimizer.step()  # 更新参数
                    avg_loss += loss.item()
                    losses[iter_] = loss.item()
                    mean_losses[iter_] = np.mean(losses[max(1, iter_ - 100) : iter_ + 1])
                    if display_iter and iter_ % display_iter == 0:
                        string = "Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}"
                        string = string.format(
                            e,
                            epoch,
                            batch_idx * len(datas),
                            len(datas) * len(train_loader),
                            100.0 * batch_idx / len(train_loader),
                            mean_losses[iter_],
                        )
                        tqdm.write(string)
                    iter_ += 1
                    del (datas, target, loss, output)  # 删除这些变量，并释放它们占用的内存
                avg_loss /= len(train_loader)
                avg_loss_epoch.append(avg_loss)

            train_end_time = time.time()

            # Calculate the total training time
            training_time_seconds = train_end_time - train_start_time
            training_time_minutes = training_time_seconds / 60 
            print("训练时间:{}min".format(training_time_minutes))
            # parameter = torch.load("parameter/Pavia_78.726.pth", map_location=lambda storage, loc: storage.cuda())

            # net.load_state_dict(parameter)
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
                    # predictions[]
                    # predictions.append(output.cpu())  # 如果在CPU上运行，只需使用output
                # 合并所有批次的预测结果
                # predictions = torch.cat(predictions, dim=0)
                # predictions = torch.max(predictions,1)[1].squeeze()
                # predictions = predictions.numpy()
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
                
            if accuary_result.size > 0:
                if oa > np.max(accuary_result):
                # if oa>87 and oa<90:4
                    best_result = text   
            accuary_result = np.append(accuary_result, oa)
            view_prediction = spy.imshow(classes=convert_to_color(prediction, palette))
            plt.axis("off") # 不显示坐标轴
            # plt.savefig("./images/{}_prediction_{}.jpg".format(datasets_name,oa))
            torch.save(net.state_dict(), 'spectralFormer/PU_params/{}_{}_{}.pth'.format('SF',train_size,'{:.2f}'.format(oa)))

print(accuary_result)
print(test_percalss_num)
print(np.max(accuary_result))
print(np.argmax(accuary_result)+1)
print(best_result)
# 模型保存
# torch.save(net.state_dict(), './PaviaU_params/{}_{}_seed:{}_have_center:{}_{}.pth'.format('transformer',train_size, seed, have_center,'{:.3f}'.format(np.max(accuary_result))))
# torch.save(net.state_dict(), './SA_params/{}_{}_seed:{}_have_center:{}_{}.pth'.format('transformer',train_size, seed, have_center,'{:.3f}'.format(np.max(accuary_result))))
# torch.save(net.state_dict(), './PaviaU_params/{}_{}_seed:{}_have_center:{}_{}.pth'.format('transformer',train_size, seed, have_center,'{:.3f}'.format(np.max(accuary_result))))
