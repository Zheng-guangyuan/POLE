import torch
import argparse
import os
from torch.utils.data import Dataset

import torchvision.transforms as T
import pickle
from Model.resnet import resnet50_baseline
import glob
import PIL.Image as Image
from tqdm import tqdm  # 导入tqdm用于显示进度条

# 创建命令行参数解析器
parser = argparse.ArgumentParser(description='abc')
# 定义命令行参数：
# --data_dir: 输入数据目录路径，包含各个slide的子文件夹
parser.add_argument('--data_dir', default='/data1/guangyuan/pole/data/patches_DTFD_1021/10.0', type=str)
# --device: 使用的设备，cuda或cpu
parser.add_argument('--device', default='cuda', type=str)
# --num_worker: 数据加载的线程数
parser.add_argument('--num_worker', default=4, type=int)
# --crop: 图像中心裁剪尺寸
parser.add_argument('--crop', default=224, type=int)
# --batch_size_v: 特征提取的批处理大小
parser.add_argument('--batch_size_v', default=64, type=int)
# --log_dir: 特征保存的输出目录
parser.add_argument('--log_dir', default='/data1/guangyuan/pole/data/features_DTFD_1021', type=str)
# --img_resize: 图像初始调整尺寸
parser.add_argument('--img_resize', default=256, type=int)

# 设置CUDA设备可见性
os.environ['CUDA_VISIBLE_DEVICES'] = "3"

def main():
    """主函数：执行特征提取的主要流程"""
    # 解析命令行参数
    args = parser.parse_args()
    
    print("开始加载特征提取模型...")
    # 加载预训练的ResNet50特征提取器并移动到指定设备
    feat_extractor = resnet50_baseline(pretrained=True).to(args.device)
    print(f"模型已加载到设备: {args.device}")

    # 定义图像标准化参数（ImageNet数据集的标准参数）
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # 定义测试时的图像变换流程
    test_transform = T.Compose([
        T.CenterCrop(args.crop),    # 中心裁剪到指定尺寸
        T.ToTensor(),               # 转换为Tensor格式
        normalize,                  # 标准化处理
    ])

    print("正在创建数据集...")
    # 创建自定义数据集实例
    all_dataset = Patch_dataset_SlideFolder_noLabel(args.data_dir, test_transform)
    print(f"数据集创建完成，共 {len(all_dataset)} 个图像块")

    # 创建数据加载器
    all_loader = torch.utils.data.DataLoader(
        all_dataset, batch_size=args.batch_size_v, shuffle=False,  # 不洗牌，保持顺序
        num_workers=args.num_worker, pin_memory=True)              # 使用多线程加载

    # 创建特征保存目录（如果不存在）
    if not os.path.exists(os.path.join(args.log_dir, 'mDATA_folder')):
        os.makedirs(os.path.join(args.log_dir, 'mDATA_folder'))
        print(f"创建特征保存目录: {os.path.join(args.log_dir, 'mDATA_folder')}")

    print("开始特征提取...")
    # 执行特征提取和保存
    extract_save_features(extractor=feat_extractor, loader=all_loader, params=args,
                          save_path=os.path.join(args.log_dir, 'mDATA_folder'))


def extract_save_features(extractor, loader, params, save_path=''):
    """
    特征提取和保存函数
    
    参数:
        extractor: 特征提取模型
        loader: 数据加载器
        params: 包含各种参数的命名空间
        save_path: 特征保存路径
    """
    # 设置模型为评估模式（关闭dropout和batchnorm的随机性）
    extractor.eval()

    # 初始化字典用于存储特征数据，结构: {slide_name: [feature_data1, feature_data2, ...]}
    mDATA = {}

    # 计算总批次数
    total_batches = len(loader)
    print(f"总批次数: {total_batches}, 批大小: {params.batch_size_v}")
    
    # 使用tqdm创建进度条
    batch_pbar = tqdm(total=total_batches, desc="提取特征", unit="batch")

    # 遍历数据加载器中的所有批次
    for idx, batchdata in enumerate(loader):
        # 获取当前批次的图像数据并移动到指定设备
        samples = batchdata['image'].to(params.device)
        slide_names = batchdata['slide_name']  # 幻灯片名称列表
        file_names = batchdata['file_name']    # 文件名列表
        
        # 提取特征（前向传播）
        feat = extractor(samples)
        
        # 将特征转换为numpy数组（从GPU移动到CPU）
        feat_np = feat.cpu().data.numpy()

        # 遍历当前批次中的每个样本
        for idx, tSlideName in enumerate(slide_names):
            # 如果该slide不在字典中，初始化一个空列表
            if tSlideName not in mDATA.keys():
                mDATA[tSlideName] = []
            
            # 获取当前样本的特征
            tFeat = feat_np[idx]
            tFileName = file_names[idx]
            
            # 构建特征数据字典
            tdata = {'feature': tFeat, 'file_name': tFileName}
            
            # 将特征数据添加到对应slide的列表中
            mDATA[tSlideName].append(tdata)
        
        # 更新进度条描述，显示当前处理的slide数量
        unique_slides = len(mDATA.keys())
        batch_pbar.set_postfix({
            '已处理slide数': unique_slides,
            '当前批次样本数': len(slide_names)
        })
        batch_pbar.update(1)  # 更新进度条

    batch_pbar.close()  # 关闭进度条

    # 如果指定了保存路径，将特征数据保存为pkl文件
    if save_path != '':
        print(f"开始保存特征到 {save_path}...")
        # 使用tqdm显示保存进度
        slide_pbar = tqdm(total=len(mDATA.keys()), desc="保存特征文件", unit="slide")
        
        for sst in mDATA.keys():
            # 为每个slide创建独立的pkl文件
            slide_save_path = os.path.join(save_path, sst+'.pkl')
            with open(slide_save_path, 'wb') as f:
                pickle.dump(mDATA[sst], f)  # 序列化保存特征数据
            
            # 更新进度条，显示当前保存的slide信息
            slide_pbar.set_postfix({'当前slide': sst, '特征数': len(mDATA[sst])})
            slide_pbar.update(1)
        
        slide_pbar.close()
        print(f"特征提取完成！共处理 {len(mDATA.keys())} 个slides")


class Patch_dataset_SlideFolder_noLabel(Dataset):
    """自定义数据集类，用于加载无标签的病理图像块"""
    
    def __init__(self, slide_dir, transform=None, img_resize=256, surfix='jpg'):
        """
        初始化数据集
        
        参数:
            slide_dir: 包含各个slide子文件夹的根目录
            transform: 图像变换 pipeline
            img_resize: 图像调整尺寸
            surfix: 图像文件后缀
        """
        self.img_resize = img_resize

        # 获取slide_dir下所有子目录（每个子目录代表一个slide）
        SlideNames = os.listdir(slide_dir)
        SlideNames = [sst for sst in SlideNames if os.path.isdir(os.path.join(slide_dir, sst))]

        # 收集所有图像块的路径
        self.patch_dirs = []
        for tslideName in SlideNames:
            # 获取该slide目录下所有指定后缀的图像文件
            tpatch_paths = glob.glob(os.path.join(slide_dir, tslideName, '*.'+surfix))
            self.patch_dirs.extend(tpatch_paths)

        self.transform = transform
        
        # 打印数据集统计信息
        print(f"数据集初始化完成:")
        print(f"  - 总slides数: {len(SlideNames)}")
        print(f"  - 总图像块数: {len(self.patch_dirs)}")
        print(f"  - 图像尺寸: {img_resize}x{img_resize}")
        print(f"  - 文件格式: {surfix}")

    def __getitem__(self, index):
        """获取单个样本"""
        # 获取图像路径
        img_dir = self.patch_dirs[index]
        
        # 打开图像并转换为RGB格式
        timage = Image.open(img_dir).convert('RGB')
        # 调整图像尺寸
        timage = timage.resize((self.img_resize, self.img_resize))

        # 从文件名解析信息
        file_name = os.path.basename(img_dir).split('.')[0]  # 去除扩展名
        tinfo = file_name.split('_')  # 假设文件名格式为: slideName_otherInfo
        slide_name = tinfo[0]         # 第一个部分为slide名称

        # 应用图像变换
        if self.transform is not None:
            timage = self.transform(timage)

        # 返回包含图像、slide名称和文件名的字典
        return {'image': timage, 'slide_name': slide_name, 'file_name': os.path.basename(img_dir)}

    def __len__(self):
        """返回数据集大小"""
        return len(self.patch_dirs)


if __name__ == '__main__':
    # 程序入口点
    main()