import pandas as pd
import numpy as np
import random
from collections import defaultdict

def create_labeled_transmil_splits(csv_path, output_path, train_frac=0.7, val_frac=0.15, test_frac=0.15, seed=42):
    """
    创建带标签的TransMIL格式静态划分CSV文件，平衡各个数据集中切片级别的正例和负例比例
    
    参数:
    csv_path: 输入CSV文件路径
    output_path: 输出CSV文件路径
    train_frac: 训练集比例
    val_frac: 验证集比例
    test_frac: 测试集比例
    seed: 随机种子
    """
    
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    
    # 读取原始数据
    df = pd.read_csv(csv_path)
    print(f"原始数据形状: {df.shape}")
    print(f"病例数量: {df['case_id'].nunique()}")
    print(f"切片数量: {df['slide_id'].nunique()}")
    
    # 统计切片级别的标签分布
    slide_label_counts = df['label'].value_counts().sort_index()
    print(f"\n切片级别标签分布:")
    for label, count in slide_label_counts.items():
        print(f"标签 {label}: {count} 个切片")
    
    # 按病例分组，确保同一个病例的所有切片在同一数据集中
    case_groups = df.groupby('case_id')
    case_data = []
    
    for case_id, group in case_groups:
        # 获取该病例的所有切片和标签
        slides = group['slide_id'].tolist()
        labels = group['label'].tolist()
        slide_info = list(zip(slides, labels))
        
        case_data.append({
            'case_id': case_id,
            'slide_info': slide_info,  # 存储(切片ID, 标签)的列表
            'num_slides': len(slides)
        })
    
    # 随机打乱病例顺序
    random.shuffle(case_data)
    
    # 初始化各数据集的切片列表
    train_slides = []
    val_slides = []
    test_slides = []
    
    # 按切片标签分组统计
    label_slides = defaultdict(list)
    for case in case_data:
        for slide_id, label in case['slide_info']:
            label_slides[label].append((slide_id, label, case['case_id']))
    
    # 对每个标签的切片进行随机打乱
    for label in label_slides:
        random.shuffle(label_slides[label])
    
    print(f"\n按标签分组的切片数量:")
    for label in sorted(label_slides.keys()):
        print(f"标签 {label}: {len(label_slides[label])} 个切片")
    
    # 为每个标签分别划分数据集，保持切片级别的比例平衡
    for label in sorted(label_slides.keys()):
        slides = label_slides[label]
        total_label_slides = len(slides)
        
        train_size = max(1, int(total_label_slides * train_frac))
        val_size = max(1, int(total_label_slides * val_frac))
        test_size = max(1, total_label_slides - train_size - val_size)
        
        # 如果test_size为0，从val_size调整
        if test_size == 0 and val_size > 1:
            val_size -= 1
            test_size = 1
        # 如果val_size为0，从train_size调整
        elif val_size == 0 and train_size > 1:
            train_size -= 1
            val_size = 1
        
        print(f"标签 {label} 切片划分: 训练集 {train_size}, 验证集 {val_size}, 测试集 {test_size}")
        
        # 划分切片
        train_slides.extend(slides[:train_size])
        val_slides.extend(slides[train_size:train_size + val_size])
        test_slides.extend(slides[train_size + val_size:train_size + val_size + test_size])
    
    # 重新组织数据：按病例分组各数据集的切片
    def organize_slides_by_case(slide_list):
        """将切片列表按病例ID重新组织"""
        case_slides = defaultdict(list)
        for slide_id, label, case_id in slide_list:
            case_slides[case_id].append((slide_id, label))
        return case_slides
    
    train_cases = organize_slides_by_case(train_slides)
    val_cases = organize_slides_by_case(val_slides)
    test_cases = organize_slides_by_case(test_slides)
    
    # 统计各数据集的切片标签分布
    def count_slide_labels(slide_list, dataset_name):
        label_counts = defaultdict(int)
        for _, label, _ in slide_list:
            label_counts[label] += 1
        print(f"{dataset_name}切片标签分布: {dict(label_counts)}")
        return label_counts
    
    train_label_counts = count_slide_labels(train_slides, "训练集")
    val_label_counts = count_slide_labels(val_slides, "验证集")
    test_label_counts = count_slide_labels(test_slides, "测试集")
    
    # 计算各数据集的正负例比例
    def calculate_balance_ratio(label_counts):
        if len(label_counts) < 2:
            return "N/A"
        labels = sorted(label_counts.keys())
        ratio = label_counts[labels[0]] / label_counts[labels[1]]
        return f"{ratio:.3f}"
    
    print(f"\n各数据集正负例比例:")
    print(f"训练集平衡比例: {calculate_balance_ratio(train_label_counts)}")
    print(f"验证集平衡比例: {calculate_balance_ratio(val_label_counts)}")
    print(f"测试集平衡比例: {calculate_balance_ratio(test_label_counts)}")
    
    # 统计病例数量
    print(f"\n病例划分情况:")
    print(f"训练集病例数: {len(train_cases)}")
    print(f"验证集病例数: {len(val_cases)}")
    print(f"测试集病例数: {len(test_cases)}")
    
    # 准备各数据集的数据（只需要切片ID和标签）
    train_slide_ids = [(slide_id, label) for slide_id, label, _ in train_slides]
    val_slide_ids = [(slide_id, label) for slide_id, label, _ in val_slides]
    test_slide_ids = [(slide_id, label) for slide_id, label, _ in test_slides]
    
    # 计算最大行数（取三个数据集的最大切片数）
    max_rows = max(len(train_slide_ids), len(val_slide_ids), len(test_slide_ids))
    print(f"最大行数: {max_rows}")
    
    # 创建新的格式：每行包含train, train_label, val, val_label, test, test_label
    trans_mil_data = []
    
    for i in range(max_rows):
        # 获取训练集数据
        if i < len(train_slide_ids):
            train_slide, train_label = train_slide_ids[i]
        else:
            train_slide, train_label = np.nan, np.nan
        
        # 获取验证集数据
        if i < len(val_slide_ids):
            val_slide, val_label = val_slide_ids[i]
        else:
            val_slide, val_label = np.nan, np.nan
        
        # 获取测试集数据
        if i < len(test_slide_ids):
            test_slide, test_label = test_slide_ids[i]
        else:
            test_slide, test_label = np.nan, np.nan
        
        row_data = {
            'train': train_slide,
            'train_label': train_label,
            'val': val_slide,
            'val_label': val_label,
            'test': test_slide,
            'test_label': test_label
        }
        trans_mil_data.append(row_data)
    
    # 创建最终的DataFrame
    trans_mil_df = pd.DataFrame(trans_mil_data)
    
    # 保存为CSV文件
    trans_mil_df.to_csv(output_path, index=True)
    
    # 统计信息
    actual_train_slides = sum(1 for slide, label in train_slide_ids if not pd.isna(slide))
    actual_val_slides = sum(1 for slide, label in val_slide_ids if not pd.isna(slide))
    actual_test_slides = sum(1 for slide, label in test_slide_ids if not pd.isna(slide))
    
    print(f"\n最终划分结果:")
    print(f"训练集切片数: {actual_train_slides}")
    print(f"验证集切片数: {actual_val_slides}")
    print(f"测试集切片数: {actual_test_slides}")
    print(f"总行数: {len(trans_mil_df)}")
    print(f"文件已保存至: {output_path}")
    
    return trans_mil_df

def verify_labeled_splits(transmil_csv_path, original_csv_path):
    """
    验证带标签划分结果是否正确
    """
    print("\n验证划分结果...")
    
    # 读取原始数据
    original_df = pd.read_csv(original_csv_path)
    original_slides = set(original_df['slide_id'])
    
    # 读取划分数据
    transmil_df = pd.read_csv(transmil_csv_path, index_col=0)
    
    # 检查是否有重复或遗漏的切片
    train_slides = set(transmil_df['train'].dropna())
    val_slides = set(transmil_df['val'].dropna())
    test_slides = set(transmil_df['test'].dropna())
    
    all_divided_slides = train_slides.union(val_slides).union(test_slides)
    
    print(f"原始切片总数: {len(original_slides)}")
    print(f"划分后切片总数: {len(all_divided_slides)}")
    print(f"是否有切片遗漏: {len(original_slides - all_divided_slides) == 0}")
    print(f"是否有额外切片: {len(all_divided_slides - original_slides) == 0}")
    
    # 检查数据集间是否有重叠
    train_val_overlap = train_slides.intersection(val_slides)
    train_test_overlap = train_slides.intersection(test_slides)
    val_test_overlap = val_slides.intersection(test_slides)
    
    print(f"训练集-验证集重叠: {len(train_val_overlap)}")
    print(f"训练集-测试集重叠: {len(train_test_overlap)}")
    print(f"验证集-测试集重叠: {len(val_test_overlap)}")
    
    # 验证标签一致性
    print(f"\n标签一致性检查:")
    
    # 检查训练集标签
    train_label_consistency = True
    for idx, row in transmil_df.iterrows():
        if not pd.isna(row['train']) and not pd.isna(row['train_label']):
            original_label = original_df[original_df['slide_id'] == row['train']]['label'].values
            if len(original_label) > 0 and original_label[0] != row['train_label']:
                print(f"训练集标签不一致: {row['train']}, 原始: {original_label[0]}, 划分文件: {row['train_label']}")
                train_label_consistency = False
    
    # 检查验证集标签
    val_label_consistency = True
    for idx, row in transmil_df.iterrows():
        if not pd.isna(row['val']) and not pd.isna(row['val_label']):
            original_label = original_df[original_df['slide_id'] == row['val']]['label'].values
            if len(original_label) > 0 and original_label[0] != row['val_label']:
                print(f"验证集标签不一致: {row['val']}, 原始: {original_label[0]}, 划分文件: {row['val_label']}")
                val_label_consistency = False
    
    # 检查测试集标签
    test_label_consistency = True
    for idx, row in transmil_df.iterrows():
        if not pd.isna(row['test']) and not pd.isna(row['test_label']):
            original_label = original_df[original_df['slide_id'] == row['test']]['label'].values
            if len(original_label) > 0 and original_label[0] != row['test_label']:
                print(f"测试集标签不一致: {row['test']}, 原始: {original_label[0]}, 划分文件: {row['test_label']}")
                test_label_consistency = False
    
    print(f"训练集标签一致性: {'通过' if train_label_consistency else '失败'}")
    print(f"验证集标签一致性: {'通过' if val_label_consistency else '失败'}")
    print(f"测试集标签一致性: {'通过' if test_label_consistency else '失败'}")
    
    # 验证标签分布
    original_label_dist = original_df['label'].value_counts().sort_index()
    print(f"\n原始数据标签分布: {dict(original_label_dist)}")
    
    return (len(train_val_overlap) == 0 and 
            len(train_test_overlap) == 0 and 
            len(val_test_overlap) == 0 and
            train_label_consistency and 
            val_label_consistency and 
            test_label_consistency)

if __name__ == "__main__":
    # 配置参数
    input_csv = "/data1/guangyuan/workspace/POLE/csv_all.csv"  # 替换为您的CSV文件路径
    output_csv = "/data1/guangyuan/workspace/POLE/TransMIL/dataset_csv/splits.csv"
    
    # 创建划分
    splits_df = create_labeled_transmil_splits(
        csv_path=input_csv,
        output_path=output_csv,
        train_frac=0.7,    # 训练集比例
        val_frac=0.15,     # 验证集比例
        test_frac=0.15,    # 测试集比例
        seed=21            # 随机种子（确保可重复性）
    )
    
    # 验证划分
    is_valid = verify_labeled_splits(output_csv, input_csv)
    
    if is_valid:
        print("\n✅ 划分验证通过！")
    else:
        print("\n❌ 划分验证失败，请检查代码！")
    
    # 显示前几行示例
    print("\n前10行示例:")
    print(splits_df.head(10))