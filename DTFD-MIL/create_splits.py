import glob
import os
import pickle
import pandas as pd

# 替换为你的.pkl文件路径
file_path = '/data1/guangyuan/workspace/features_CONCH_1030/'
save_path = '/data1/guangyuan/workspace/features_CONCH_1030_DTFD'
# 替换为你的csv文件路径
df = pd.read_csv('/data1/guangyuan/workspace/POLE/DTFD-MIL/splits_balance.csv')

# 初始化字典
mDATA_train = {}
mDATA_val = {}
mDATA_test = {}

# 获取所有pkl文件路径
feature_paths = glob.glob(os.path.join(file_path, '*.pkl'))

# 打印文件数量用于调试
print(f"找到 {len(feature_paths)} 个pkl文件")

# 从CSV文件中提取slide名称到对应的集合中
train_slides = set()
val_slides = set()
test_slides = set()

# 处理train列
for slide_name in df['train'].dropna():
    train_slides.add(str(slide_name).strip())

# 处理val列
for slide_name in df['val'].dropna():
    val_slides.add(str(slide_name).strip())

# 处理test列
for slide_name in df['test'].dropna():
    test_slides.add(str(slide_name).strip())

print(f"训练集slide数量: {len(train_slides)}")
print(f"验证集slide数量: {len(val_slides)}")
print(f"测试集slide数量: {len(test_slides)}")

# 处理每个pkl文件
for slide_file in feature_paths:
    try:
        with open(slide_file, 'rb') as file:
            data = pickle.load(file)
        
        print(f'shape of data from {slide_file}: {data.shape}')
        file_name = os.path.basename(slide_file).split('.')[0]
        
        # 根据您的文件命名格式，可能需要调整slide名称的提取方式
        # 假设pkl文件名格式为 "slide_name.pkl" 或 "slide_name_other_info.pkl"
        # 这里我们直接使用文件名（去掉扩展名）作为slide名称
        slide_name = file_name.split('_')[0]
        
        # 检查slide_name是否在对应的集合中
        matched = False
        if slide_name in train_slides:
            mDATA_train[slide_name] = data
            print(f"添加到训练集: {slide_name}")
            matched = True
        if slide_name in val_slides:
            mDATA_val[slide_name] = data
            print(f"添加到验证集: {slide_name}")
            matched = True
        if slide_name in test_slides:
            mDATA_test[slide_name] = data
            print(f"添加到测试集: {slide_name}")
            matched = True
            
        if not matched:
            print(f"未匹配到任何集合: {slide_name}")
            
    except Exception as e:
        print(f"处理文件 {slide_file} 时出错: {e}")

# 打印各集合最终大小
print(f"训练集最终大小: {len(mDATA_train)}")
print(f"验证集最终大小: {len(mDATA_val)}")
print(f"测试集最终大小: {len(mDATA_test)}")

# 保存结果
if not os.path.exists(save_path):
    os.makedirs(save_path)

slide_train_save_path = os.path.join(save_path, 'train.pkl')
slide_val_save_path = os.path.join(save_path, 'val.pkl')
slide_test_save_path = os.path.join(save_path, 'test.pkl')

try:
    with open(slide_train_save_path, 'wb') as f:
        pickle.dump(mDATA_train, f)
    print(f"训练集已保存到: {slide_train_save_path}, 文件大小: {os.path.getsize(slide_train_save_path)} 字节")
    
    with open(slide_val_save_path, 'wb') as f:
        pickle.dump(mDATA_val, f)
    print(f"验证集已保存到: {slide_val_save_path}, 文件大小: {os.path.getsize(slide_val_save_path)} 字节")
    
    with open(slide_test_save_path, 'wb') as f:
        pickle.dump(mDATA_test, f)
    print(f"测试集已保存到: {slide_test_save_path}, 文件大小: {os.path.getsize(slide_test_save_path)} 字节")
    
except Exception as e:
    print(f"保存文件时出错: {e}")