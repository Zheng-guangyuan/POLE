import pandas as pd
import numpy as np
import random

def create_transmil_splits(csv_path, output_path, train_frac=0.7, val_frac=0.15, test_frac=0.15, seed=42):
    """
    创建TransMIL格式的静态划分CSV文件
    
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
    
    # 按病例分组，确保同一个病例的所有切片在同一数据集中
    case_groups = df.groupby('case_id')
    case_data = []
    
    for case_id, group in case_groups:
        # 获取该病例的所有切片和标签
        slides = group['slide_id'].tolist()
        labels = group['label'].tolist()
        
        # 检查标签是否一致（同一个病例应该有相同的标签）
        if len(set(labels)) > 1:
            print(f"警告: 病例 {case_id} 有多个标签: {set(labels)}，使用多数投票")
            # 使用多数投票确定标签
            label = max(set(labels), key=labels.count)
        else:
            label = labels[0]
        
        case_data.append({
            'case_id': case_id,
            'slides': slides,
            'label': label,
            'num_slides': len(slides)
        })
    
    # 转换为DataFrame
    case_df = pd.DataFrame(case_data)
    
    # 随机打乱病例顺序
    case_df = case_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # 计算各数据集的大小
    total_cases = len(case_df)
    train_size = int(total_cases * train_frac)
    val_size = int(total_cases * val_frac)
    test_size = total_cases - train_size - val_size
    
    print(f"\n划分情况:")
    print(f"训练集病例数: {train_size}")
    print(f"验证集病例数: {val_size}")
    print(f"测试集病例数: {test_size}")
    
    # 划分数据集
    train_cases = case_df.iloc[:train_size]
    val_cases = case_df.iloc[train_size:train_size + val_size]
    test_cases = case_df.iloc[train_size + val_size:]
    
    # 计算最大行数（取三个数据集的最大长度）
    max_train_slides = sum(len(case['slides']) for _, case in train_cases.iterrows())
    max_val_slides = sum(len(case['slides']) for _, case in val_cases.iterrows())
    max_test_slides = sum(len(case['slides']) for _, case in test_cases.iterrows())
    
    max_rows = max(max_train_slides, max_val_slides, max_test_slides)
    print(f"最大行数: {max_rows}")
    
    # 创建TransMIL格式的DataFrame
    trans_mil_data = []
    
    # 准备各数据集的数据
    train_slides = []
    for _, case in train_cases.iterrows():
        train_slides.extend([(slide, case['label']) for slide in case['slides']])
    
    val_slides = []
    for _, case in val_cases.iterrows():
        val_slides.extend([(slide, case['label']) for slide in case['slides']])
    
    test_slides = []
    for _, case in test_cases.iterrows():
        test_slides.extend([(slide, case['label']) for slide in case['slides']])
    
    # 填充到最大行数
    while len(train_slides) < max_rows:
        train_slides.append((np.nan, np.nan))
    
    while len(val_slides) < max_rows:
        val_slides.append((np.nan, np.nan))
    
    while len(test_slides) < max_rows:
        test_slides.append((np.nan, np.nan))
    
    # 创建每一行的数据
    for i in range(max_rows):
        train_slide, train_label = train_slides[i] if i < len(train_slides) else (np.nan, np.nan)
        val_slide, val_label = val_slides[i] if i < len(val_slides) else (np.nan, np.nan)
        test_slide, test_label = test_slides[i] if i < len(test_slides) else (np.nan, np.nan)
        
        row_data = {
            'train': train_slide,
            'val': val_slide,
            'test': test_slide,
        }
        trans_mil_data.append(row_data)
    
    # 创建最终的DataFrame
    trans_mil_df = pd.DataFrame(trans_mil_data)
    
    # 保存为CSV文件
    trans_mil_df.to_csv(output_path, index=True)
    
    # 统计信息
    actual_train_slides = sum(1 for slide, label in train_slides if not pd.isna(slide))
    actual_val_slides = sum(1 for slide, label in val_slides if not pd.isna(slide))
    actual_test_slides = sum(1 for slide, label in test_slides if not pd.isna(slide))
    
    print(f"\n最终划分结果:")
    print(f"训练集切片数: {actual_train_slides}")
    print(f"验证集切片数: {actual_val_slides}")
    print(f"测试集切片数: {actual_test_slides}")
    print(f"总行数: {len(trans_mil_df)}")
    print(f"文件已保存至: {output_path}")
    
    return trans_mil_df

def verify_splits(transmil_csv_path, original_csv_path):
    """
    验证划分结果是否正确
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
    
    return len(train_val_overlap) == 0 and len(train_test_overlap) == 0 and len(val_test_overlap) == 0

if __name__ == "__main__":
    # 配置参数
    input_csv = "/data1/guangyuan/pole/results/output.csv"  # 替换为您的CSV文件路径
    output_csv = "/data1/guangyuan/pole/DTFD-MIL/splits.csv"
    
    # 创建划分
    splits_df = create_transmil_splits(
        csv_path=input_csv,
        output_path=output_csv,
        train_frac=0.7,    # 训练集比例
        val_frac=0.15,     # 验证集比例
        test_frac=0.15,    # 测试集比例
        seed=21            # 随机种子（确保可重复性）
    )
    
    # 验证划分
    is_valid = verify_splits(output_csv, input_csv)
    
    if is_valid:
        print("\n✅ 划分验证通过！")
    else:
        print("\n❌ 划分验证失败，请检查代码！")
    
    # 显示前几行示例
    print("\n前10行示例:")
    print(splits_df.head(10))