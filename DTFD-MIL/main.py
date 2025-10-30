import torch
# 为多进程共享张量设置策略，避免某些系统的FileDescriptor限制问题
torch.multiprocessing.set_sharing_strategy('file_system')
# 打开自动求导异常检测（在调试时有助于定位反向传播中的NaN/inf等问题）
torch.autograd.set_detect_anomaly(True)

import argparse
import json
import pandas as pd
import os
from torch.utils.tensorboard import SummaryWriter
import pickle
import random
from Model.Attention import Attention_Gated as Attention
from Model.Attention import Attention_with_Classifier
from utils import get_cam_1d
import torch.nn.functional as F
from Model.network import Classifier_1fc, DimReduction
import numpy as np
from utils import eval_metric
from tqdm import tqdm  # 添加tqdm用于进度条显示

# 创建命令行参数解析器
parser = argparse.ArgumentParser(description='abc')
testMask_dir = ''  # 指向Camelyon测试集掩码位置

# 定义所有命令行参数
parser.add_argument('--name', default='10-30-0212-conch', type=str)  # 实验名称

parser.add_argument('--EPOCH', default=300, type=int)  # 总训练轮数
parser.add_argument('--epoch_step', default='[30]', type=str)  # 学习率衰减的epoch节点
parser.add_argument('--device', default='cuda:1', type=str)  # 训练设备
parser.add_argument('--isPar', default=False, type=bool)  # 是否使用数据并行
parser.add_argument('--log_dir', default='./debug_log', type=str)  # 日志文件路径
parser.add_argument('--train_show_freq', default=100, type=int)  # 训练日志显示频率
parser.add_argument('--droprate', default='0', type=float)  # dropout率
parser.add_argument('--droprate_2', default='0', type=float)  # 第二层dropout率
parser.add_argument('--lr', default=1e-5, type=float)  # 学习率
parser.add_argument('--weight_decay', default=1e-4, type=float)  # 权重衰减
parser.add_argument('--lr_decay_ratio', default=0.2, type=float)  # 学习率衰减比例
parser.add_argument('--batch_size', default=1, type=int)  # 训练batch大小
parser.add_argument('--batch_size_v', default=1, type=int)  # 验证/测试batch大小
parser.add_argument('--num_workers', default=4, type=int)  # 数据加载工作进程数
parser.add_argument('--num_cls', default=2, type=int)  # 分类类别数
# 数据路径参数
parser.add_argument('--mDATA0_dir_train0', default='/data1/guangyuan/workspace/features_CONCH_1029_DTFD/train.pkl', type=str)  # 训练集
parser.add_argument('--mDATA0_dir_val0', default='/data1/guangyuan/workspace/features_CONCH_1029_DTFD/val.pkl', type=str)  # 验证集
parser.add_argument('--mDATA_dir_test0', default='/data1/guangyuan/workspace/features_CONCH_1029_DTFD/test.pkl', type=str)  # 测试集
parser.add_argument('--label_file', default='/data1/guangyuan/workspace/POLE/labels.csv', type=str)  # 标签文件
# 模型结构参数
parser.add_argument('--numGroup', default=4, type=int)  # 分组数量
parser.add_argument('--total_instance', default=16, type=int)  # 总实例数
parser.add_argument('--numGroup_test', default=4, type=int)  # 测试时分组数量
parser.add_argument('--total_instance_test', default=16, type=int)  # 测试时总实例数
parser.add_argument('--mDim', default=512, type=int)  # 特征维度
parser.add_argument('--grad_clipping', default=5, type=float)  # 梯度裁剪阈值
parser.add_argument('--isSaveModel', action='store_false')  # 是否保存模型
parser.add_argument('--numLayer_Res', default=0, type=int)  # ResNet层数
parser.add_argument('--num_MeanInference', default=1, type=int)  # 平均推理次数
parser.add_argument('--distill_type', default='MaxMinS', type=str)  # 蒸馏类型: MaxMinS, MaxS, AFS

# 设置随机种子保证可重复性
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
random.seed(42)

def main():
    torch.autograd.set_detect_anomaly(True)
    params = parser.parse_args()  # 解析命令行参数
    epoch_step = json.loads(params.epoch_step)  # 解析学习率衰减节点
    writer = SummaryWriter(os.path.join(params.log_dir, 'LOG', params.name))  # 创建TensorBoard写入器

    print("=" * 60)
    print("开始初始化模型和训练过程...")
    print(f"实验名称: {params.name}")
    print(f"总训练轮数: {params.EPOCH}")
    print(f"设备: {params.device}")
    print("=" * 60)

    in_chn = 1024  # 输入特征通道数

    print("正在加载模型组件...")
    # 初始化模型组件
    classifier = Classifier_1fc(params.mDim, params.num_cls, params.droprate).to(params.device)  # 分类器
    attention = Attention(params.mDim).to(params.device)  # 注意力机制
    dimReduction = DimReduction(in_chn, params.mDim, numLayer_Res=params.numLayer_Res).to(params.device)  # 维度缩减
    attCls = Attention_with_Classifier(L=params.mDim, num_cls=params.num_cls, droprate=params.droprate_2).to(params.device)  # 带分类器的注意力

    # 如果启用数据并行，包装模型
    if params.isPar:
        classifier = torch.nn.DataParallel(classifier)
        attention = torch.nn.DataParallel(attention)
        dimReduction = torch.nn.DataParallel(dimReduction)
        attCls = torch.nn.DataParallel(attCls)
        print("已启用数据并行训练")
    
    torch.autograd.set_detect_anomaly(True)

    # 定义损失函数
    ce_cri = torch.nn.CrossEntropyLoss(reduction='none').to(params.device)

    # 创建日志目录和文件
    if not os.path.exists(params.log_dir):
        os.makedirs(params.log_dir)
    log_dir = os.path.join(params.log_dir, f'log_{params.name}.txt')
    save_dir = os.path.join(params.log_dir, f'best_model_{params.name}.pth')
    z = vars(params).copy()
    with open(log_dir, 'a') as f:
        f.write(json.dumps(z))
    log_file = open(log_dir, 'a')

    print("正在加载数据集...")
    # 加载训练、验证和测试数据
    with open(params.mDATA0_dir_train0, 'rb') as f:
        mDATA_train = pickle.load(f)
    with open(params.mDATA0_dir_val0, 'rb') as f:
        mDATA_val = pickle.load(f)
    with open(params.mDATA_dir_test0, 'rb') as f:
        mDATA_test = pickle.load(f)
    
    # 加载标签字典
    label_dict = load_label(params.label_file)

    # 重组数据格式
    SlideNames_train, FeatList_train, Label_train = reOrganize_mDATA(mDATA_train, label_dict)
    SlideNames_val, FeatList_val, Label_val = reOrganize_mDATA(mDATA_val, label_dict)
    SlideNames_test, FeatList_test, Label_test = reOrganize_mDATA(mDATA_test, label_dict)

    print_log(f'训练数据: {len(SlideNames_train)}个slides', log_file)
    print_log(f'验证数据: {len(SlideNames_val)}个slides', log_file)
    print_log(f'测试数据: {len(SlideNames_test)}个slides', log_file)

    # 设置优化器参数
    trainable_parameters = []
    trainable_parameters += list(classifier.parameters())
    trainable_parameters += list(attention.parameters())
    trainable_parameters += list(dimReduction.parameters())

    # 创建AdamW优化器（两个优化器分别优化不同部分）
    optimizer_adam0 = torch.optim.AdamW(trainable_parameters, lr=params.lr, weight_decay=params.weight_decay)
    optimizer_adam1 = torch.optim.AdamW(attCls.parameters(), lr=params.lr, weight_decay=params.weight_decay)

    # 创建学习率调度器
    scheduler0 = torch.optim.lr_scheduler.MultiStepLR(optimizer_adam0, epoch_step, gamma=params.lr_decay_ratio)
    scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer_adam1, epoch_step, gamma=params.lr_decay_ratio)

    # 初始化最佳指标
    best_auc = 0
    best_epoch = -1
    test_auc = 0

    print("\n开始训练过程...")
    # 创建总进度条
    epoch_pbar = tqdm(total=params.EPOCH, desc="总训练进度", unit="epoch")
    
    for ii in range(params.EPOCH):
        epoch_pbar.set_description(f"Epoch {ii+1}/{params.EPOCH}")
        
        # 打印当前学习率
        for param_group in optimizer_adam1.param_groups:
            curLR = param_group['lr']
            print_log(f'当前学习率: {curLR}', log_file)

        # 训练阶段
        train_attention_preFeature_DTFD(classifier=classifier, dimReduction=dimReduction, attention=attention, UClassifier=attCls, 
                                       mDATA_list=(SlideNames_train, FeatList_train, Label_train), ce_cri=ce_cri,
                                       optimizer0=optimizer_adam0, optimizer1=optimizer_adam1, epoch=ii, params=params, 
                                       f_log=log_file, writer=writer, numGroup=params.numGroup, 
                                       total_instance=params.total_instance, distill=params.distill_type)
        
        # 验证阶段
        print_log(f'>>>>>>>>>>> 验证阶段 Epoch: {ii}', log_file)
        auc_val = test_attention_DTFD_preFeat_MultipleMean(classifier=classifier, dimReduction=dimReduction, attention=attention,
                                                           UClassifier=attCls, mDATA_list=(SlideNames_val, FeatList_val, Label_val), 
                                                           criterion=ce_cri, epoch=ii, params=params, f_log=log_file, 
                                                           writer=writer, numGroup=params.numGroup_test, 
                                                           total_instance=params.total_instance_test, distill=params.distill_type)
        print_log(' ', log_file)
        
        # 测试阶段
        print_log(f'>>>>>>>>>>> 测试阶段 Epoch: {ii}', log_file)
        tauc = test_attention_DTFD_preFeat_MultipleMean(classifier=classifier, dimReduction=dimReduction, attention=attention,
                                                        UClassifier=attCls, mDATA_list=(SlideNames_test, FeatList_test, Label_test), 
                                                        criterion=ce_cri, epoch=ii, params=params, f_log=log_file, 
                                                        writer=writer, numGroup=params.numGroup_test, 
                                                        total_instance=params.total_instance_test, distill=params.distill_type)
        print_log(' ', log_file)

        # 保存最佳模型（在训练后期）
        if ii > int(params.EPOCH*0.6):
            if auc_val > best_auc:
                best_auc = auc_val
                best_epoch = ii
                test_auc = tauc
                if params.isSaveModel:
                    tsave_dict = {
                        'classifier': classifier.state_dict(),
                        'dim_reduction': dimReduction.state_dict(),
                        'attention': attention.state_dict(),
                        'att_classifier': attCls.state_dict()
                    }
                    torch.save(tsave_dict, save_dir)
                    print_log(f'模型已保存到: {save_dir}', log_file)

            print_log(f'当前最佳测试AUC: {test_auc}, 来自epoch {best_epoch}', log_file)

        # 更新学习率
        scheduler0.step()
        scheduler1.step()
        epoch_pbar.update(1)
        epoch_pbar.set_postfix({
            '最佳AUC': f'{best_auc:.4f}',
            '当前验证AUC': f'{auc_val:.4f}',
            '当前测试AUC': f'{tauc:.4f}'
        })

    epoch_pbar.close()
    print_log(f'\n训练完成! 最终最佳测试AUC: {test_auc}, 来自epoch {best_epoch}', log_file)
    log_file.close()

# 加载标签数据
def load_label(label_file_path):
    """从CSV文件加载标签数据"""
    if not label_file_path:
        return None

    print(f"正在加载标签文件: {label_file_path}")
    label_dict = {}
    df = pd.read_csv(label_file_path)
    for _, row in df.iterrows():
        label_dict[row['case_id']] = row['label']

    print(f"标签加载完成，共 {len(label_dict)} 个样本")
    return label_dict

def test_attention_DTFD_preFeat_MultipleMean(mDATA_list, classifier, dimReduction, attention, UClassifier, epoch, criterion=None, params=None, f_log=None, writer=None, numGroup=3, total_instance=3, distill='MaxMinS'):
    """测试函数 - 使用多均值推理的DTFD方法"""
    
    # 设置模型为评估模式
    classifier.eval()
    attention.eval()
    dimReduction.eval()
    UClassifier.eval()

    SlideNames, FeatLists, Label = mDATA_list
    instance_per_group = total_instance // numGroup

    # 初始化损失记录器
    test_loss0 = AverageMeter()
    test_loss1 = AverageMeter()

    # 初始化预测和真实标签存储
    gPred_0 = torch.FloatTensor().to(params.device)  # 第一层预测
    gt_0 = torch.LongTensor().to(params.device)      # 第一层真实标签
    gPred_1 = torch.FloatTensor().to(params.device)  # 第二层预测
    gt_1 = torch.LongTensor().to(params.device)      # 第二层真实标签

    with torch.no_grad():  # 禁用梯度计算
        numSlides = len(SlideNames)
        numIter = numSlides // params.batch_size_v
        tIDX = list(range(numSlides))

        # 创建测试进度条
        test_pbar = tqdm(total=numIter, desc="测试进度", unit="batch", leave=False)
        
        for idx in range(numIter):
            # 获取当前batch的数据
            tidx_slide = tIDX[idx * params.batch_size_v:(idx + 1) * params.batch_size_v]
            slide_names = [SlideNames[sst] for sst in tidx_slide]
            tlabel = [Label[sst] for sst in tidx_slide]
            label_tensor = torch.LongTensor(tlabel).to(params.device)
            batch_feat = [FeatLists[sst].to(params.device) for sst in tidx_slide]

            for tidx, tfeat in enumerate(batch_feat):
                tslideName = slide_names[tidx]
                tslideLabel = label_tensor[tidx].unsqueeze(0)
                midFeat = dimReduction(tfeat)  # 维度缩减

                AA = attention(midFeat, isNorm=False).squeeze(0)  # 计算注意力权重 N

                allSlide_pred_softmax = []  # 存储所有推理的预测

                # 多次均值推理
                for jj in range(params.num_MeanInference):
                    # 随机打乱特征索引并分组
                    feat_index = list(range(tfeat.shape[0]))
                    random.shuffle(feat_index)
                    index_chunk_list = np.array_split(np.array(feat_index), numGroup)
                    index_chunk_list = [sst.tolist() for sst in index_chunk_list]

                    slide_d_feat = []   # 蒸馏特征
                    slide_sub_preds = []  # 子预测
                    slide_sub_labels = []  # 子标签

                    # 处理每个特征组
                    for tindex in index_chunk_list:
                        slide_sub_labels.append(tslideLabel)
                        idx_tensor = torch.LongTensor(tindex).to(params.device)
                        tmidFeat = midFeat.index_select(dim=0, index=idx_tensor)

                        # 计算注意力特征
                        tAA = AA.index_select(dim=0, index=idx_tensor)
                        tAA = torch.softmax(tAA, dim=0)  # 归一化注意力权重
                        tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)  # 加权特征 n x fs
                        tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)  # 聚合特征 1 x fs

                        tPredict = classifier(tattFeat_tensor)  # 第一层预测 1 x 2
                        slide_sub_preds.append(tPredict)

                        # 获取CAM（类激活图）用于特征选择
                        patch_pred_logits = get_cam_1d(classifier, tattFeats.unsqueeze(0)).squeeze(0)  # cls x n
                        patch_pred_logits = torch.transpose(patch_pred_logits, 0, 1)  # n x cls
                        patch_pred_softmax = torch.softmax(patch_pred_logits, dim=1)  # n x cls

                        # 根据预测置信度排序
                        _, sort_idx = torch.sort(patch_pred_softmax[:, -1], descending=True)

                        # 根据蒸馏类型选择特征
                        if distill == 'MaxMinS':
                            # 选择置信度最高和最低的特征
                            topk_idx_max = sort_idx[:instance_per_group].long()
                            topk_idx_min = sort_idx[-instance_per_group:].long()
                            topk_idx = torch.cat([topk_idx_max, topk_idx_min], dim=0)
                            d_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)
                            slide_d_feat.append(d_inst_feat)
                        elif distill == 'MaxS':
                            # 只选择置信度最高的特征
                            topk_idx_max = sort_idx[:instance_per_group].long()
                            topk_idx = topk_idx_max
                            d_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)
                            slide_d_feat.append(d_inst_feat)
                        elif distill == 'AFS':
                            # 使用注意力加权特征
                            slide_d_feat.append(tattFeat_tensor)

                    # 合并所有组特征
                    slide_d_feat = torch.cat(slide_d_feat, dim=0)
                    slide_sub_preds = torch.cat(slide_sub_preds, dim=0)
                    slide_sub_labels = torch.cat(slide_sub_labels, dim=0)

                    # 记录第一层结果
                    gPred_0 = torch.cat([gPred_0, slide_sub_preds], dim=0)
                    gt_0 = torch.cat([gt_0, slide_sub_labels], dim=0)
                    loss0 = criterion(slide_sub_preds, slide_sub_labels).mean()
                    test_loss0.update(loss0.item(), numGroup)

                    # 第二层预测
                    gSlidePred = UClassifier(slide_d_feat)
                    allSlide_pred_softmax.append(torch.softmax(gSlidePred, dim=1))

                # 多推理结果平均
                allSlide_pred_softmax = torch.cat(allSlide_pred_softmax, dim=0)
                allSlide_pred_softmax = torch.mean(allSlide_pred_softmax, dim=0).unsqueeze(0)
                gPred_1 = torch.cat([gPred_1, allSlide_pred_softmax], dim=0)
                gt_1 = torch.cat([gt_1, tslideLabel], dim=0)

                # 计算第二层损失
                loss1 = F.nll_loss(allSlide_pred_softmax, tslideLabel)
                test_loss1.update(loss1.item(), 1)

            test_pbar.update(1)
            test_pbar.set_postfix({
                '当前slide': slide_names[0][:10] + '...' if len(slide_names[0]) > 10 else slide_names[0],
                'Loss0': f'{test_loss0.avg:.4f}',
                'Loss1': f'{test_loss1.avg:.4f}'
            })

        test_pbar.close()

    # 计算评估指标
    gPred_0 = torch.softmax(gPred_0, dim=1)
    gPred_0 = gPred_0[:, -1]  # 取正类概率
    gPred_1 = gPred_1[:, -1]  # 取正类概率

    # 评估第一层性能
    macc_0, mprec_0, mrecal_0, mspec_0, mF1_0, auc_0 = eval_metric(gPred_0, gt_0)
    # 评估第二层性能
    macc_1, mprec_1, mrecal_1, mspec_1, mF1_1, auc_1 = eval_metric(gPred_1, gt_1)

    print_log(f'  第一层 - 准确率: {macc_0:.4f}, 精确率: {mprec_0:.4f}, 召回率: {mrecal_0:.4f}, 特异度: {mspec_0:.4f}, F1: {mF1_0:.4f}, AUC: {auc_0:.4f}', f_log)
    print_log(f'  第二层 - 准确率: {macc_1:.4f}, 精确率: {mprec_1:.4f}, 召回率: {mrecal_1:.4f}, 特异度: {mspec_1:.4f}, F1: {mF1_1:.4f}, AUC: {auc_1:.4f}', f_log)

    # 记录到TensorBoard
    writer.add_scalar(f'auc_0 ', auc_0, epoch)
    writer.add_scalar(f'auc_1 ', auc_1, epoch)

    return auc_1  # 返回第二层AUC作为主要指标

def train_attention_preFeature_DTFD(mDATA_list, classifier, dimReduction, attention, UClassifier, optimizer0, optimizer1, epoch, ce_cri=None, params=None, f_log=None, writer=None, numGroup=3, total_instance=3, distill='MaxMinS'):
    """训练函数 - DTFD方法"""
    
    SlideNames_list, mFeat_list, Label_dict = mDATA_list

    # 设置模型为训练模式
    classifier.train()
    dimReduction.train()
    attention.train()
    UClassifier.train()

    instance_per_group = total_instance // numGroup

    # 初始化损失记录器
    Train_Loss0 = AverageMeter()  # 第一层损失
    Train_Loss1 = AverageMeter()  # 第二层损失

    numSlides = len(SlideNames_list)
    numIter = numSlides // params.batch_size

    tIDX = list(range(numSlides))
    random.shuffle(tIDX)  # 随机打乱训练顺序

    # 创建训练进度条
    train_pbar = tqdm(total=numIter, desc="训练进度", unit="batch", leave=False)

    for idx in range(numIter):
        # 获取当前batch
        tidx_slide = tIDX[idx * params.batch_size:(idx + 1) * params.batch_size]

        tslide_name = [SlideNames_list[sst] for sst in tidx_slide]
        tlabel = [Label_dict[sst] for sst in tidx_slide]
        label_tensor = torch.LongTensor(tlabel).to(params.device)

        for tidx, (tslide, slide_idx) in enumerate(zip(tslide_name, tidx_slide)):
            tslideLabel = label_tensor[tidx].unsqueeze(0)

            slide_pseudo_feat = []   # 伪特征用于第二层
            slide_sub_preds = []     # 第一层预测
            slide_sub_labels = []    # 第一层标签

            tfeat_tensor = mFeat_list[slide_idx]
            tfeat_tensor = tfeat_tensor.to(params.device)

            # 随机分组特征
            feat_index = list(range(tfeat_tensor.shape[0]))
            random.shuffle(feat_index)
            index_chunk_list = np.array_split(np.array(feat_index), numGroup)
            index_chunk_list = [sst.tolist() for sst in index_chunk_list]

            # 处理每个特征组
            for tindex in index_chunk_list:
                slide_sub_labels.append(tslideLabel)
                subFeat_tensor = torch.index_select(tfeat_tensor, dim=0, index=torch.LongTensor(tindex).to(params.device))
                tmidFeat = dimReduction(subFeat_tensor)  # 维度缩减
                tAA = attention(tmidFeat).squeeze(0)  # 注意力权重
                tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)  # 加权特征 n x fs
                tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)  # 聚合特征 1 x fs
                tPredict = classifier(tattFeat_tensor)  # 第一层预测 1 x 2
                slide_sub_preds.append(tPredict)

                # 获取CAM用于特征选择
                patch_pred_logits = get_cam_1d(classifier, tattFeats.unsqueeze(0)).squeeze(0)  # cls x n
                patch_pred_logits = torch.transpose(patch_pred_logits, 0, 1)  # n x cls
                patch_pred_softmax = torch.softmax(patch_pred_logits, dim=1)  # n x cls

                # 根据置信度排序选择特征
                _, sort_idx = torch.sort(patch_pred_softmax[:, -1], descending=True)
                topk_idx_max = sort_idx[:instance_per_group].long()  # 高置信度
                topk_idx_min = sort_idx[-instance_per_group:].long()  # 低置信度
                topk_idx = torch.cat([topk_idx_max, topk_idx_min], dim=0)

                # 准备不同蒸馏类型的特征
                MaxMin_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)  # MaxMin特征
                max_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx_max)  # 仅Max特征
                af_inst_feat = tattFeat_tensor  # 注意力特征

                # 根据蒸馏类型选择特征
                if distill == 'MaxMinS':
                    slide_pseudo_feat.append(MaxMin_inst_feat)
                elif distill == 'MaxS':
                    slide_pseudo_feat.append(max_inst_feat)
                elif distill == 'AFS':
                    slide_pseudo_feat.append(af_inst_feat)

            # 合并所有组特征
            slide_pseudo_feat = torch.cat(slide_pseudo_feat, dim=0)  # numGroup x fs

            ## 优化第一层
            slide_sub_preds = torch.cat(slide_sub_preds, dim=0)  # numGroup x 2
            slide_sub_labels = torch.cat(slide_sub_labels, dim=0)  # numGroup
            loss0 = ce_cri(slide_sub_preds, slide_sub_labels).mean()  # 第一层损失
            
            optimizer0.zero_grad()
            loss0.backward(retain_graph=True)  # 保留计算图用于第二层反向传播
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(dimReduction.parameters(), params.grad_clipping)
            torch.nn.utils.clip_grad_norm_(attention.parameters(), params.grad_clipping)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), params.grad_clipping)
            optimizer0.step()

            ## 优化第二层
            gSlidePred = UClassifier(slide_pseudo_feat)  # 第二层预测
            loss1 = ce_cri(gSlidePred, tslideLabel).mean()  # 第二层损失
            
            optimizer1.zero_grad()
            loss1.backward()
            torch.nn.utils.clip_grad_norm_(UClassifier.parameters(), params.grad_clipping)
            optimizer1.step()

            # 更新损失记录
            Train_Loss0.update(loss0.item(), numGroup)
            Train_Loss1.update(loss1.item(), 1)

        train_pbar.update(1)
        train_pbar.set_postfix({
            'Loss0': f'{Train_Loss0.avg:.4f}',
            'Loss1': f'{Train_Loss1.avg:.4f}'
        })

        # 定期显示训练信息
        if idx % params.train_show_freq == 0:
            tstr = 'epoch: {} idx: {}'.format(epoch, idx)
            tstr += f' 第一层损失: {Train_Loss0.avg:.4f}, 第二层损失: {Train_Loss1.avg:.4f} '
            print_log(tstr, f_log)

    train_pbar.close()
    # 记录损失到TensorBoard
    writer.add_scalar(f'train_loss_0 ', Train_Loss0.avg, epoch)
    writer.add_scalar(f'train_loss_1 ', Train_Loss1.avg, epoch)

class AverageMeter(object):
    """计算和存储平均值和当前值的实用类"""

    def __init__(self):
        self.reset()

    def reset(self):
        """重置所有统计量"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """更新统计量"""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def print_log(tstr, f):
    """打印日志到文件和控制台"""
    f.write('\n')
    f.write(tstr)
    print(tstr)

def reOrganize_mDATA(mDATA, label_dict=None):
    """重组数据格式，将原始数据转换为模型可用的格式"""
    SlideNames = []
    FeatList = []
    Label = []
    
    print("正在重组数据...")
    
    for slide_name in mDATA.keys():
        SlideNames.append(slide_name)

        # 从slide名称提取case_id并获取标签
        case_id = slide_name.split('-')[0]
        if label_dict is not None and case_id in label_dict:
            label = label_dict[case_id]
        
        Label.append(label)

        # 处理每个slide的特征
        patch_data_list = mDATA[slide_name]
        featGroup = []
        for tpatch in patch_data_list:
            tfeat = torch.from_numpy(tpatch['feature'])
            featGroup.append(tfeat.unsqueeze(0))
        featGroup = torch.cat(featGroup, dim=0)  # 合并所有特征
        FeatList.append(featGroup)
    
    # 验证类别分布
    positive_count = sum(1 for label in Label if label == 1)
    negative_count = sum(1 for label in Label if label == 0)
    
    print(f"数据重组完成: 总样本 {len(Label)}, 正样本 {positive_count}, 负样本 {negative_count}")
    
    # 检查类别平衡
    if positive_count == 0 or negative_count == 0:
        print(f"警告: 数据集中只有一种类别! 正样本: {positive_count}, 负样本: {negative_count}")

    return SlideNames, FeatList, Label

if __name__ == "__main__":
    main()