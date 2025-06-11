#---------------------------------------------------#
#   跟utils_lp一样，只是sample_gt不一样，这里标记样本分为了训练集，验证集，测试集, 未标记样本集
#---------------------------------------------------#
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import sklearn.model_selection
import torch
from sklearn.metrics import confusion_matrix
import itertools
import scipy.io as sio
import os
from sklearn import preprocessing
import math
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import normalize
import torch.nn.functional as F
from operator import truediv

def loadIndianPinesData():
        data_path = os.path.join(os.getcwd(),'/home/lxl/project/datasets')
        dataname = 3
        if dataname == 1:
            name = "Indian_pines"
            rgb_bands = (43, 21, 11)
            data = sio.loadmat(os.path.join(data_path, 'Indian_pines_corrected.mat'))['indian_pines_corrected']
            labels = sio.loadmat(os.path.join(data_path, 'Indian_pines_gt.mat'))['indian_pines_gt']
            label_values = [
            "Alfalfa",
            "Corn-notill",
            "Corn-mintill",
            "Corn",
            "Grass-pasture",
            "Grass-trees",
            "Grass-pasture-mowed",
            "Hay-windrowed",
            "Oats",
            "Soybean-notill",
            "Soybean-mintill",
            "Soybean-clean",
            "Wheat",
            "Woods",
            "Buildings-Grass-Trees-Drives",
            "Stone-Steel-Towers",
        ]
        elif dataname == 2:     
            name = "KSC"   
            rgb_bands = (43, 21, 11)
            data = sio.loadmat(os.path.join(data_path, 'KSC.mat'))['KSC']
            labels = sio.loadmat(os.path.join(data_path, 'KSC_gt.mat'))['KSC_gt'] 
            label_values = [
            "Scrub",
            "Willow swamp",
            "Cabbage palm hammock",
            "Cabbage palm/oak hammock",
            "Slash pine",
            "Oak/broadleaf hammock",
            "Hardwood swamp",
            "Graminoid marsh",
            "Spartina marsh",
            "Cattail marsh",
            "Salt marsh",
            "Mud flats",
            "Water",
        ]
        elif dataname == 3:    
            name = "Salinas"
            rgb_bands = (43, 21, 11)
            data = sio.loadmat(os.path.join(data_path, 'Salinas_corrected.mat'))['salinas_corrected']
            labels = sio.loadmat(os.path.join(data_path, 'Salinas_gt.mat'))['salinas_gt']
            label_values = [
            "Brocoli_green_weeds_1",
            "Brocoli_green_weeds_2",
            "Fallow",
            "Fallow_rough_plow",
            "Fallow_smooth",
            "Stubble",
            "Celery",
            "Grapes_untrained",
            "Soil_vinyard_develop",
            "Corn_senesced_green_weeds",
            "Lettuce_romaine_4wk",
            "Lettuce_romaine_5wk",
            "Lettuce_romaine_6wk",
            "Lettuce_romaine_7wk",
            "Vinyard_untrained",
            "Vinyard_vertical_trellis",
        ]
        elif dataname == 4:    
            name = "PaviaU"
            rgb_bands = (55, 41, 12)
            data = sio.loadmat(os.path.join(data_path, 'PaviaU.mat'))['paviaU']
            labels = sio.loadmat(os.path.join(data_path, 'PaviaU_gt.mat'))['paviaU_gt'] 
            label_values = [
            "Asphalt",
            "Meadows",
            "Gravel",
            "Trees",
            "Painted metal sheets",
            "Bare Soil",
            "Bitumen",
            "Self-Blocking Bricks",
            "Shadows",
        ]
        elif dataname == 5:       
            # Houston2013
            name = "Houston2013"
            data = sio.loadmat(os.path.join(data_path, 'Houston.mat'))['Houston']
            labels = sio.loadmat(os.path.join(data_path, 'Houston_gt.mat'))['Houston_gt']  
            label_values = [  # 15
                "Healthy Grass",
                "Stressed Grass",
                "Synthetis Grass",
                "Tree",
                "Soil",
                "Water",
                "Residential",
                "Commercial",
                "Road",
                "Highyway",
                "Railway",
                "Parking Lot 1",
                "Parking Lot 2",
                "Tennis Court",
                "Running Track"]
        elif dataname == 6:
            name = "Botswana"   
            rgb_bands = (75, 33, 15)
            data = sio.loadmat(os.path.join(data_path, 'Botswana.mat'))['Botswana']
            labels = sio.loadmat(os.path.join(data_path, 'Botswana_gt.mat'))['Botswana_gt'] 
            label_values = [  # 14
                "Water",
                "Hippo grass",
                "Floodplain grasses 1",
                "Floodplain grasses 2",
                "Reeds",
                "Riparian",
                "Firescar",
                "Island interior",
                "Acacia woodlands",
                "Acacia shrublands",
                "Acacia grasslands",
                "Short mopane",
                "Mixed mopane",
                "Exposed soils",
            ]     
        elif dataname == 7:
            name = "Pavia Centre"   
            data = sio.loadmat(os.path.join(data_path, 'Pavia.mat'))['pavia']
            labels = sio.loadmat(os.path.join(data_path, 'Pavia_gt.mat'))['pavia_gt'] 
            label_values = ['Water', 'Trees','Asphalt','Self-blocking Bricks', 'Bitumen', 'Tiles', 'Shadows', 'Meadows','Bare Soil']                                                             
        return data, labels, label_values, name

def standartizeData(X):
        newX = np.reshape(X, (-1, X.shape[2]))  # 这个是所有通道上进行标准化
        scaler = preprocessing.StandardScaler().fit(newX) 
        newX = scaler.transform(newX)
        newX = np.reshape(newX, (X.shape[0],X.shape[1],X.shape[2]))
        return newX, scaler
def padWithZeros(X, margin):
        newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
        x_offset = margin
        y_offset = margin
        newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
        return newX

def createPatches(X, y, windowSize=5, removeZeroLabels =False):# non False ;sample  True
        margin = int((windowSize - 1) / 2)
        zeroPaddedX = padWithZeros(X, margin=windowSize//2)
        patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
        patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
        patchIndex = 0
        for r in range(margin, zeroPaddedX.shape[0] - margin):
            for c in range(margin, zeroPaddedX.shape[1] - margin):
                patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]   
                patchesData[patchIndex, :, :, :] = patch
                patchesLabels[patchIndex] = y[r-margin, c-margin]
                patchIndex = patchIndex + 1
        if removeZeroLabels:
            patchesData = patchesData[patchesLabels>0,:,:,:]
            patchesLabels = patchesLabels[patchesLabels>0]
            patchesLabels -= 1     #为了one-hot要减一
        return patchesData, patchesLabels

def applyPCA(X, numComponents=75):
        newX = np.reshape(X, (-1, X.shape[2]))
        pca = PCA(n_components=numComponents, whiten=True)
        newX = pca.fit_transform(newX)
        newX = np.reshape(newX, (X.shape[0],X.shape[1], numComponents))
        return newX, pca

def sample_gt(gt, train_size, patch_size):
    indices = np.nonzero(gt)
    # gt是数组类型的  numpy.ndarray
    # indices 是元组类型，元组里有两个元素，元素是数组类型，第一个数组是[0,0,...,1095] 第二个数组是[161,162,....,569]
    # zip(*indices)将两个数组的对应位置的两个元素组成一个元组(如（0，161）),表示gt里类别为非0的位置索引
    X = list(zip(*indices))  # x,y features  X是索引元组组合的列表
    # 这里将gt的非0类别拉成一维数组，gt[indices]是找到indices位置的元素，如果这里indices是(0,161)那么gt[indices]是1
    y = gt[indices].ravel()  # classes  gt[indices] [1 1 1 ... 2 2 2]
    train_gt = np.zeros_like(gt)  # gt的shape是(1096, 715)
    test_gt = np.zeros_like(gt)
    val_gt = np.zeros_like(gt) 
    unlabeled_gt = np.zeros_like(gt)
    train_indices, test_indices, val_indices, unlabel_indices= [], [], [], []

    for c in np.unique(gt):
        if c == 0:
            continue
        indices = np.nonzero(gt == c)
        X = list(zip(*indices)) # x,y features
        train, temp = sklearn.model_selection.train_test_split(X, train_size=train_size,  random_state=3407)
        if(len(temp)>=220):   
            val, temp = sklearn.model_selection.train_test_split(temp, train_size=20, random_state=3407)
        else:
            val = []
        if(len(temp)>=230):
            test, unlabel = sklearn.model_selection.train_test_split(temp, train_size=200, random_state=3407)
        else:
            test = temp
            unlabel = []
        print(len(train), len(test))
        train_indices += train    # train :[(73, 98), (72, 96), (71, 98), (71, 99), (68, 96), (66, 96), (71, 97), (70, 100), (70, 96), (64, 96)]
        test_indices += test 
        val_indices += val
        unlabel_indices += unlabel
    train_indices_list = [list(t) for t in zip(*train_indices)]   # 
    test_indices_list = [list(t) for t in zip(*test_indices)]  
    val_indices_list = [list(t) for t in zip(*val_indices)]  
    unlabel_indices_list = [list(t) for t in zip(*unlabel_indices)]  
    gt = torch.from_numpy(gt)
    train_gt = torch.from_numpy(train_gt)
    test_gt = torch.from_numpy(test_gt)
    val_gt = torch.from_numpy(val_gt)
    unlabeled_gt = torch.from_numpy(unlabeled_gt)
    train_gt[train_indices_list] = gt[train_indices_list]
    test_gt[test_indices_list] = gt[test_indices_list]
    val_gt[val_indices_list] = gt[val_indices_list]
    unlabeled_gt[unlabel_indices_list] = gt[unlabel_indices_list]
    # 查找非零元素的索引
    nonzero_indices = torch.nonzero(train_gt)
    nonzero_val = torch.nonzero(val_gt)
    nonzero_test = torch.nonzero(test_gt)
    # 统计非零元素的数量
    nonzero_count = nonzero_indices.size(0)
    # 打印非零元素的数量
    print("训练集数量:", nonzero_count)
    print("验证集数量:", nonzero_val.size(0))
    print("除去训练集邻域的测试样本之前的数量:", nonzero_test.size(0))
    # 将训练样本patch里除中心像素以外的像素点从测试集里删除
    test_gt_delete = test_gt.clone() # test_gt_delete是删减掉训练集的patch里的像素
    test_gt_len= torch.nonzero(test_gt)
    # print("长度", len(test_gt_len))
    for row, col in train_indices:
    # 获取将每个训练样本的坐标-> 获取patch的首坐标-> 遍历patch的每个坐标-> 将测试集的坐标=0
    # 获取patch的首坐标
        a = row-patch_size//2  # //" 是整数除法运算符。它用于执行除法操作并返回结果的整数部分，而不考虑小数部分
        b = col-patch_size//2
        coordinates = [(a + j, b + i) for i in range(patch_size) for j in range(patch_size)]
        for i,j in coordinates:
            if i>0 and i<145 and j>0 and j<145:
                test_gt_delete[i][j]=0
    # 145 * 145 = 21025
    # 查找非零元素的索引
    nonzero_indices = torch.nonzero(test_gt_delete)
    # 统计非零元素的数量
    nonzero_count = nonzero_indices.size(0)
    # 打印非零元素的数量
    print("之后的数量:", nonzero_count)
    # return train_gt, val_gt, test_gt, test_gt_delete
    return train_gt, val_gt, test_gt, unlabeled_gt


def convert_to_color(arr_2d, palette):
    """Convert an array of labels to RGB color-encoded image.

    Args:
        arr_2d: int 2D array of labels
        palette: dict of colors used (label number -> RGB tuple)

    Returns:
        arr_3d: int 2D images of color-encoded labels in RGB format

    """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)
    if palette is None:
        raise Exception("Unknown color palette")

    for c, i in palette.items():
        m = arr_2d == c
         # c是类别  i是rgb
        arr_3d[m] = i
    return arr_3d
    
def val(net, data_loader, device, supervision="full"):
    # TODO : fix me using metrics()
    accuracy, total = 0.0, 0.0
    for batch_idx, (data, target) in enumerate(data_loader):
        with torch.no_grad():
            # Load the data into the GPU if required
            data, target = data.to(device), target.to(device)
            output, _, _= net(data)  # 这是transformer 的
            # _, _, _, output= net(data)
            _, output = torch.max(output, dim=1) # 找到每个样本的最大值及其索引  output是索引也是预测的类别
            # view() 是 PyTorch 中的一个方法，用于调整张量（tensor）的形状，即改变张量的维度
            for out, pred in zip(output.view(-1), target.view(-1)):   # view(-1) 表示将张量展平为一个一维的张量，而不关心该张量的具体形状
                accuracy += out.item() == pred.item()
                total += 1
    return accuracy / total

def sliding_window(image, patch_size):
    p = patch_size//2
    # data_pad = np.pad(image, ((p, p), (p, p), (0, 0)), mode="symmetric")
    w = patch_size
    h = patch_size
    W, H = image.shape[:2]
    for x in range(0, image.shape[0]-patch_size):
        for y in range(0, image.shape[1]-patch_size):
            yield image[x:x + w, y:y + h], x, y, w, h

def grouper(n, iterable):
    """ Browse an iterable by grouping n elements by n elements.

    Args:
        n: int, size of the groups
        iterable: the iterable to Browse
    Yields:
        chunk of n elements from the iterable

    """
    it = iter(iterable)   # 将可迭代对象 iterable 转换为一个迭代器，并将其赋值给变量 it。这样可以通过迭代器来逐个访问 iterable 中的元素
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk

def metrics(prediction, target, test_percalss_num, n_classes):
    """Compute and print metrics (accuracy, confusion matrix and F1 scores).

    Args:
        prediction: list of predicted labels
        target: list of target labels
        ignored_labels (optional): list of labels to ignore, e.g. 0 for undef
        n_classes (optional): number of classes, max(target) by default
    Returns:
        accuracy, F1 score by class, confusion matrix
    """
    # 获取一个多维数组或张量 target 的前两个轴的维度 target.shape  (610, 340)
    # 第一步是 mask[gt==0] = True，它使用布尔索引来将 gt 数组中值为 0 的位置对应的 mask 数组的元素设置为 True。
    # 第二步是 mask = ~mask，它对 mask 数组进行按位取反的操作，将 True 转换为 False，将 False 转换为 True。
    # 最后一步是 print(mask[mask])，它打印出 mask 数组中值为 True 的元素。
    ignored_mask = np.zeros(target.shape[:2], dtype=bool)
    ignored_mask[target == 0] = True
    ignored_mask = ~ignored_mask  # 这段代码使用了位运算符 ~ 对布尔数组 ignored_mask 进行按位取反的操作。
    # print(ignored_mask.shape)  # (610, 340)
    target = target[ignored_mask]  # 打印出target位置上为非0的元素
    prediction = prediction[ignored_mask]  # 打印出target位置上为非0的元素
    # print(len(target))  # 42766
    # print(len(prediction))  # 42766
    results = {}
    n_classes = np.max(target) + 1 if n_classes is None else n_classes   # n_classes = 10
    cm = confusion_matrix(
        target,
        prediction,
        labels=range(1, n_classes))

    results["Confusion matrix"] = cm

    # Compute global accuracy
    total = np.sum(cm)
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)

    results["OA"] = accuracy

    # AA
    each_acc, aa = AA_andEachClassAccuracy(cm)
    results["AA"] = aa*100
    # Compute F1 score
    F1scores = np.zeros(len(cm))  #
    Class_Acc = np.zeros(len(cm))  #
    for i in range(len(cm)):
        denominator = (np.sum(cm[i, :]) + np.sum(cm[:, i]))
        if denominator == 0:
            F1 = 0.
        else:
            F1 = 2. * cm[i, i] / denominator
            class_acc = cm[i, i] / test_percalss_num[i] * 1.
            class_acc *= 100 
        F1scores[i] = F1
        Class_Acc[i] = class_acc
    results["F1 scores"] = F1scores
    results["class acc"] = Class_Acc

    # Compute kappa coefficient
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / \
        float(total * total)
    kappa = (pa - pe) / (1 - pe)
    results["Kappa"] = kappa*100

    return results

def AA_andEachClassAccuracy(confusion_matrix):
        list_diag = np.diag(confusion_matrix)
        list_raw_sum = np.sum(confusion_matrix, axis=1)
        each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
        average_acc = np.mean(each_acc)
        return each_acc, average_acc

def show_results(results, label_values=None):
    text = ""
    cm = results["Confusion matrix"]
    accuracy = results["OA"]
    AA = results["AA"]
    class_acc = results["class acc"]
    F1scores = results["F1 scores"]
    kappa = results["Kappa"]

    text += "Confusion matrix :\n"
    text += str(cm)
    text += "---\n"
    text += "OA : {:.02f}%\n".format(accuracy)
    text += "---\n"
    text += "AA : {:.02f}%\n".format(AA)
    text += "---\n"
    # text += "F1 scores :\n"
    # for label, score in zip(label_values, F1scores):
    #     text += "\t{}: {:.03f}\n".format(label, score)
    text += "class acc :\n" 
    for label, score in zip(label_values, class_acc):
        text += "\t{}: {:.02f}\n".format(label, score)
    text += "---\n"
    text += "Kappa: {:.02f}%\n".format(kappa)
    print(text)
    return accuracy, text

def softmax(x):
    # 计算指数部分
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    # 计算分母部分，即所有指数的和
    sum_exp_x = np.sum(exp_x, axis=-1, keepdims=True)
    # 计算每个类别的概率
    probabilities = exp_x / sum_exp_x
    probabilities = np.round(probabilities, 3)
    return probabilities

# 分为训练集、验证集 剩下为测试集
def sample_gt_1(gt, train_size, patch_size):
    indices = np.nonzero(gt)
    X = list(zip(*indices))  # x,y features  X是索引元组组合的列表
    # 这里将gt的非0类别拉成一维数组，gt[indices]是找到indices位置的元素，如果这里indices是(0,161)那么gt[indices]是1
    y = gt[indices].ravel()  # classes  gt[indices] [1 1 1 ... 2 2 2]
    train_gt = np.zeros_like(gt)  # gt的shape是(1096, 715)
    test_gt = np.zeros_like(gt)
    val_gt = np.zeros_like(gt) 
    train_indices, test_indices, val_indices= [], [], []
    for c in np.unique(gt):
        if c == 0:
            continue
        indices = np.nonzero(gt == c)
        X = list(zip(*indices)) # x,y features
        if(len(X)*train_size<5):
            train, temp = sklearn.model_selection.train_test_split(X, train_size=5,  random_state=3407)
        else:
            train, temp = sklearn.model_selection.train_test_split(X, train_size=train_size,  random_state=3407)
        if(len(temp)>=220):   
            val, temp = sklearn.model_selection.train_test_split(temp, train_size=20, random_state=3407)
        else:
            val = []
        test = temp
        print(len(train), len(test))
        train_indices += train    # train :[(73, 98), (72, 96), (71, 98), (71, 99), (68, 96), (66, 96), (71, 97), (70, 100), (70, 96), (64, 96)]
        test_indices += test 
        val_indices += val
    train_indices_list = [list(t) for t in zip(*train_indices)]   # 
    test_indices_list = [list(t) for t in zip(*test_indices)]  
    val_indices_list = [list(t) for t in zip(*val_indices)]  
    gt = torch.from_numpy(gt)
    train_gt = torch.from_numpy(train_gt)
    test_gt = torch.from_numpy(test_gt)
    val_gt = torch.from_numpy(val_gt)
    train_gt[train_indices_list] = gt[train_indices_list]
    test_gt[test_indices_list] = gt[test_indices_list]
    val_gt[val_indices_list] = gt[val_indices_list]
    # 查找非零元素的索引
    nonzero_indices = torch.nonzero(train_gt)
    nonzero_val = torch.nonzero(val_gt)
    nonzero_test = torch.nonzero(test_gt)
    # 统计非零元素的数量
    nonzero_count = nonzero_indices.size(0)
    # 打印非零元素的数量
    print("训练集数量:", nonzero_count)
    print("验证集数量:", nonzero_val.size(0))
    print("除去训练集邻域的测试样本之前的数量:", nonzero_test.size(0))
    # 将训练样本patch里除中心像素以外的像素点从测试集里删除
    test_gt_delete = test_gt.clone() # test_gt_delete是删减掉训练集的patch里的像素
    test_gt_len= torch.nonzero(test_gt)
    # print("长度", len(test_gt_len))
    for row, col in train_indices:
    # 获取将每个训练样本的坐标-> 获取patch的首坐标-> 遍历patch的每个坐标-> 将测试集的坐标=0
    # 获取patch的首坐标
        a = row-patch_size//2  # //" 是整数除法运算符。它用于执行除法操作并返回结果的整数部分，而不考虑小数部分
        b = col-patch_size//2
        coordinates = [(a + j, b + i) for i in range(patch_size) for j in range(patch_size)]
        for i,j in coordinates:
            if i>0 and i<145 and j>0 and j<145:
                test_gt_delete[i][j]=0
    # 145 * 145 = 21025
    # 查找非零元素的索引
    nonzero_indices = torch.nonzero(test_gt_delete)
    # 统计非零元素的数量
    nonzero_count = nonzero_indices.size(0)
    # 打印非零元素的数量
    print("之后的数量:", nonzero_count)
    # return train_gt, val_gt, test_gt, test_gt_delete
    return train_gt, val_gt, test_gt

def spectral_clustering(X, n_clusters, n_neighbors=10, gamma=1.0):
    # 构建相似度图
    affinity_matrix = pairwise_kernels(X, metric='rbf', gamma=gamma)

    # 构建邻接矩阵
    adjacency_matrix = kneighbors_graph(X, n_neighbors, mode='connectivity', include_self=True)

    # 构建拉普拉斯矩阵
    laplacian_matrix = normalize(adjacency_matrix, norm='l1', axis=1)
    laplacian_matrix = np.eye(X.shape[0]) - laplacian_matrix

    # 计算前几个非零特征向量
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian_matrix)
    indices = np.argsort(eigenvalues)
    eigenvectors = eigenvectors[:, indices[:n_clusters]]

    # 使用K均值对特征向量进行聚类
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(eigenvectors)

    return labels


def custom_loss(output, target, confidence):
    # 假设output是模型的主要输出，target是真实标签，confidence是置信度分支的输出

    # 计算主要任务的损失，例如交叉熵
    main_loss = F.cross_entropy(output, target)
    confidence = torch.sigmoid(confidence)
    # 根据置信度调整损失
    adjusted_loss = main_loss * (1-confidence)

    # 返回加权后的总损失
    return adjusted_loss.mean()


def data_preprocessing(data):
    '''
    1. normalization
    2. pca
    3. spectral filter
    data: [h, w, spectral]
    '''
    # 1. normalization
    norm_data = np.zeros(data.shape)
    for i in range(data.shape[2]):
        input_max = np.max(data[:,:,i])  # 每个光谱维度上进行处理
        input_min = np.min(data[:,:,i])
        norm_data[:,:,i] = (data[:,:,i]-input_min)/(input_max-input_min)


    # 2. pca
    # pca_num = self.data_param.get('pca', 0)
    # if pca_num > 0:
    #     print('before pca')
    #     pca_data = self.applyPCA(norm_data, int(self.data_param['pca']))
    #     norm_data = pca_data
    #     # print("pca之后的data.shape", norm_data.shape)  # (512, 217, 900)
    #     print('after pca')
    # 3. spectral filter    # spectracl_size = 900
    # if self.spectracl_size > 0: # 按照给定的spectral size截取数据
    #     norm_data = norm_data[:,:,:self.spectracl_size]
    return norm_data


def sample_gt_2(gt, train_size):
    indices = np.nonzero(gt)   # gt是数组类型的  numpy.ndarray
    X = list(zip(*indices))  # x,y features  X是索引元组组合的列表
    # 这里将gt的非0类别拉成一维数组，gt[indices]是找到indices位置的元素，如果这里indices是(0,161)那么gt[indices]是1
    y = gt[indices].ravel()  # classes  gt[indices] [1 1 1 ... 2 2 2]
    # y.shape  (148152,)
    # print(gt[indices].shape)  # (148152,)
    train_gt = np.zeros_like(gt)  # gt的shape是(1096, 715)
    test_gt = np.zeros_like(gt)
    val_gt = np.zeros_like(gt)
    test_percalss_num = []
    if train_size > 1:
        train_size = int(train_size)
    train_indices, test_indices, val_indices = [], [], []
    for c in np.unique(gt):
        if c == 0:
            continue
        indices = np.nonzero(gt == c)
        X = list(zip(*indices)) # x,y features
        print(len(X))
        # if len(X)<=20:
        #     train, test = sklearn.model_selection.train_test_split(X, train_size=10)
        # else:
        #     train, test = sklearn.model_selection.train_test_split(X, train_size=train_size)
        train, test = sklearn.model_selection.train_test_split(X, train_size=train_size)
        # if(len(test)>train_size):
        #     val, test = sklearn.model_selection.train_test_split(test, train_size=train_size)
        train_indices += train
        # val_indices += val
        test_indices += test
        test_percalss_num.append(len(test))
    train_coords = [(x, y) for x, y in train_indices if x < gt.shape[0] and y < gt.shape[1]]
    for x, y in train_coords:
        train_gt[x, y] = gt[x, y]
    # val_coords = [(x, y) for x, y in val_indices if x < gt.shape[0] and y < gt.shape[1]]
    # for x, y in val_coords:
    #     val_gt[x, y] = gt[x, y]
    test_coords = [(x, y) for x, y in test_indices if x < gt.shape[0] and y < gt.shape[1]]
    for x, y in test_coords:
        test_gt[x, y] = gt[x, y]
    nonzero_indices = np.nonzero(train_gt)
    # nonzero_val = np.nonzero(val_gt)
    nonzero_test = np.nonzero(test_gt)
    # 统计非零元素的数量
    nonzero_count = len(nonzero_indices[0])
    # 打印非零元素的数量
    print("训练集数量:", nonzero_count)
    # print("验证集数量:", len(nonzero_val[0]))
    print("测试集数量:", len(nonzero_test[0]))
    # nonzero_count = np.count_nonzero(train_gt)
    # print(nonzero_count)
    # print(train_gt)
    # print(train_gt.shape)
    # exit()
    # return test_percalss_num, train_gt,val_gt,test_gt
    return test_percalss_num, train_gt, test_gt

# 示例用法
# X 是数据矩阵，每行代表一个样本
# n_clusters 是聚类的簇数
# 可根据实际情况调整其他参数





