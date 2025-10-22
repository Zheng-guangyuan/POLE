import os
import numpy as np
import openslide
import sdpc
import glob
from multiprocessing import Pool, Value, Lock
from skimage.filters import threshold_otsu
import cv2
import PIL.Image as Image
import traceback
import time

####======================================    User Configuration
num_thread = 4
patch_dimension_level = 1    ## 0: 40x, 1: 20x
patch_level_list = [1]  #[1,2,2]
psize = 256
stride = 256
psize_list = [256]
tissue_mask_threshold = 0.9
mask_dimension_level = 5

# 图像信息含量过滤参数
enable_information_filter = True  # 是否启用信息含量过滤
information_threshold = 3.0  # 信息熵阈值（一般范围3-6，越高要求越严格）
min_std_threshold = 15.0  # 最小标准差阈值（用于检测空白图像）

slides_folder_dir = 'E:\SYSUXY'
slide_paths = glob.glob(os.path.join(slides_folder_dir, '*.sdpc'))  # change the surfix '.tif' to other if necessary
save_folder_dir = 'D:\AAAAny\TEMP\patches_DTFD_1021'

# 创建失败记录文件
failed_slides_file = os.path.join(save_folder_dir, 'failed_slides.txt')
os.makedirs(save_folder_dir, exist_ok=True)
####======================================

# 创建锁对象用于线程安全的文件写入
file_lock = Lock()

def log_failed_slide(slide_path, error_message):
    """记录失败的slide到文件"""
    with file_lock:
        with open(failed_slides_file, 'a') as f:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{timestamp} - {slide_path} - Error: {error_message}\n")


def calculate_patch_information(patch_pil):
    """
    计算patch的信息含量，用于过滤低信息量（如空白、单色）的patch
    
    参数:
        patch_pil: PIL Image对象，RGB格式
    
    返回:
        tuple: (entropy, std_dev) - 信息熵和标准差
    """
    # 转换为numpy数组
    patch_array = np.array(patch_pil)
    
    # 1. 计算标准差 (检测是否为空白/单色图像)
    # 如果图像几乎是单一颜色，标准差会很低
    std_dev = np.std(patch_array)
    
    # 2. 计算信息熵 (Shannon entropy)
    # 熵越高，图像信息量越大
    # 将RGB图像转换为灰度图进行熵计算
    gray = cv2.cvtColor(patch_array, cv2.COLOR_RGB2GRAY)
    
    # 计算灰度直方图
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    
    # 归一化直方图得到概率分布
    hist = hist.astype(float)
    hist = hist / hist.sum()
    
    # 计算熵（排除概率为0的bin）
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))
    
    return entropy, std_dev


def is_informative_patch(patch_pil, entropy_threshold=3.0, std_threshold=15.0):
    """
    判断patch是否包含足够的信息量
    
    参数:
        patch_pil: PIL Image对象
        entropy_threshold: 信息熵阈值，默认3.0（典型范围3-6）
        std_threshold: 标准差阈值，默认15.0
    
    返回:
        bool: True表示patch有足够信息量，False表示应该过滤掉
    """
    entropy, std_dev = calculate_patch_information(patch_pil)
    
    # 两个条件都要满足：
    # 1. 信息熵要高于阈值（图像有足够的灰度变化）
    # 2. 标准差要高于阈值（图像不是单一颜色）
    is_informative = (entropy >= entropy_threshold) and (std_dev >= std_threshold)
    
    return is_informative

def get_roi_bounds(tslide, mask_level=3, cls_kernel=50, open_kernal=30):
    """
    获取组织区域掩模和边界框
    """
    try:
        # 自动调整 mask_level
        if mask_level >= tslide.level_count:
            mask_level = tslide.level_count - 1
        
        # 读取低倍图像
        subSlide = tslide.read_region((0, 0), mask_level, tslide.level_dimensions[mask_level])
        subSlide_np = np.array(subSlide)

        # 转 HSV
        hsv = cv2.cvtColor(subSlide_np, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Otsu 阈值
        hthresh = threshold_otsu(h)
        sthresh = threshold_otsu(s)
        vthresh = threshold_otsu(v)

        minhsv = np.array([hthresh, sthresh, 70], np.uint8)
        maxhsv = np.array([180, 255, vthresh], np.uint8)
        mask = cv2.inRange(hsv, minhsv, maxhsv)

        # 闭运算去孔
        close_kernel = np.ones((cls_kernel, cls_kernel), dtype=np.uint8)
        mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
        # 开运算去噪
        open_kernel = np.ones((open_kernal, open_kernal), dtype=np.uint8)
        mask_opened = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, open_kernel)

        contours_result = cv2.findContours(mask_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours_result[-2] if len(contours_result) >= 2 else []
        boundingBox = [cv2.boundingRect(c) for c in contours]
        # boundingBox = [bbox for bbox in boundingBox if bbox[2] > 150 and bbox[3] > 150]

        print(f'找到 {len(boundingBox)} 个组织区域')
        return mask_opened, boundingBox
    except Exception as e:
        print(f"获取ROI边界时发生错误: {e}")
        return np.nan, np.nan
    

def Extract_Patch_From_Slide_STRIDE(tslide, tissue_mask, patch_save_folder, patch_level, mask_level, patch_stride, patch_size, threshold, level_list=[1], patch_size_list=[256], patch_surfix='jpg'):
    try:
        assert patch_level == level_list[0]
        assert patch_size == patch_size_list[0]

        mask_h, mask_w = tissue_mask.shape
        print(f'组织掩模形状: {tissue_mask.shape} (mask_level={mask_level})')

        # mask 单位像素代表的是 mask_level 级别的像素
        # mask_patch_size: 在 mask 级别下每个 patch 的边长（像素）
        mask_patch_size = patch_size // (2 ** (mask_level - patch_level))
        if mask_patch_size <= 0:
            raise ValueError('计算得到的 mask_patch_size <= 0，请检查级别与尺寸参数')

        mask_patch_area = mask_patch_size * mask_patch_size
        mask_stride = patch_stride // (2 ** (mask_level - patch_level))
        if mask_stride <= 0:
            mask_stride = 1

        print(f'掩模级别 patch_size={mask_patch_size}, stride={mask_stride}')

        tslide_name = os.path.basename(patch_save_folder)
        num_error = 0
        total_patches = 0
        saved_patches = 0

        # 用于从 mask 级别坐标映射到基础像素坐标（level 0）：
        # scale_mask_to_l0 = 2 ** mask_level
        # 更通用的做法：level_dimensions 提供每级别的尺寸，计算 scale = level0_dim / level_mask_dim
        # 但 openslide 的标准金字塔每级缩放为 2 倍，所以上面的 2**mask_level 是常见做法。
        # 为安全起见，从 tslide.level_dimensions 计算映射比例（基于宽）
        level_dims = tslide.level_dimensions
        # 目标在 level 0 的像素单位映射因子（mask_level -> level0）
        scale_mask_to_l0_w = level_dims[0][0] / float(level_dims[mask_level][0])
        scale_mask_to_l0_h = level_dims[0][1] / float(level_dims[mask_level][1])

        # 遍历 mask 网格
        grid_w = (mask_w - mask_patch_size) // mask_stride + 1
        grid_h = (mask_h - mask_patch_size) // mask_stride + 1

        for iy in range(grid_h):
            for ix in range(grid_w):
                mw = ix * mask_stride
                mh = iy * mask_stride
                # 只有当完整 patch 在 mask 内时才考虑
                if (mw + mask_patch_size) <= mask_w and (mh + mask_patch_size) <= mask_h:
                    total_patches += 1
                    mask_patch = tissue_mask[mh:mh + mask_patch_size, mw:mw + mask_patch_size]
                    # mask_patch 里非零像素比例
                    mRatio = float(np.sum(mask_patch > 0)) / mask_patch_area

                    if mRatio >= threshold:
                        # mask_patch 的中心（以 mask_level 像素坐标）
                        center_mask_x = mw + mask_patch_size // 2
                        center_mask_y = mh + mask_patch_size // 2

                        # 将中心映射到 level0 像素坐标
                        center_l0_x = int(round(center_mask_x * scale_mask_to_l0_w))
                        center_l0_y = int(round(center_mask_y * scale_mask_to_l0_h))

                        # 对每个要求的保存级别，计算对应的 read_region 左上角
                        for sstLevel, tSize in zip(level_list, patch_size_list):
                            try:
                                tsave_folder = getFolder_name(patch_save_folder, sstLevel, tSize)
                                os.makedirs(tsave_folder, exist_ok=True)

                                # 目标级别相对于 level0 的缩放： scale_l0_to_target = level_dims[sstLevel][0] / level_dims[0][0]
                                # 但 read_region 接受 (x, y) 在 level 0 的坐标且第二个参数是要读取的级别，
                                # 所以我们应该直接给出 level0 坐标并传入 sstLevel。
                                # 计算要读取的左上角（在 level0 坐标下）
                                half_tsize_l0_w = int(round((tSize / float(level_dims[sstLevel][0])) * level_dims[0][0] / 2.0))
                                half_tsize_l0_h = int(round((tSize / float(level_dims[sstLevel][1])) * level_dims[0][1] / 2.0))
                                # 上面的计算比较复杂且容易出错；更直观的做法：
                                # - 将中心从 mask_level -> target_level 直接计算缩放因子：
                                scale_mask_to_target_x = level_dims[sstLevel][0] / float(level_dims[mask_level][0])
                                scale_mask_to_target_y = level_dims[sstLevel][1] / float(level_dims[mask_level][1])
                                center_target_x = int(round(center_mask_x * scale_mask_to_target_x))
                                center_target_y = int(round(center_mask_y * scale_mask_to_target_y))
                                # read_region 的位置参数是基于目标级别的像素坐标的左上角，
                                # 但是 openslide.read_region 的第一个参数 (location) 是 level 0 的坐标，第二个是 level。
                                # 因此把 center_target（在 target level 像素）映回 level0：
                                scale_target_to_l0_x = level_dims[0][0] / float(level_dims[sstLevel][0])
                                scale_target_to_l0_y = level_dims[0][1] / float(level_dims[sstLevel][1])
                                center_l0_for_target_x = int(round(center_target_x * scale_target_to_l0_x))
                                center_l0_for_target_y = int(round(center_target_y * scale_target_to_l0_y))

                                # 左上角在 level0 坐标
                                tl_l0_x = center_l0_for_target_x - int(round(tSize * scale_target_to_l0_x / 2.0))
                                tl_l0_y = center_l0_for_target_y - int(round(tSize * scale_target_to_l0_y / 2.0))

                                # 边界检查：确保 tl 在 [0, level0_dim - read_region_size_in_level0]
                                read_w_l0 = int(round(tSize * scale_target_to_l0_x))
                                read_h_l0 = int(round(tSize * scale_target_to_l0_y))

                                # 裁剪左上角和尺寸以保证在 level0 范围内
                                lvl0_w, lvl0_h = level_dims[0]
                                if tl_l0_x < 0:
                                    tl_l0_x = 0
                                if tl_l0_y < 0:
                                    tl_l0_y = 0
                                if tl_l0_x + read_w_l0 > lvl0_w:
                                    tl_l0_x = max(0, lvl0_w - read_w_l0)
                                if tl_l0_y + read_h_l0 > lvl0_h:
                                    tl_l0_y = max(0, lvl0_h - read_h_l0)

                                # 使用 read_region: location 是 level0 坐标, level=sstLevel, size=(tSize, tSize)
                                # 但 openslide.read_region 的 location 对于所请求的 level 会被内部缩放，
                                # 因此我们应该将 location 提供为与 sstLevel 对齐的 level0 坐标，
                                # 即传入 (tl_l0_x, tl_l0_y), sstLevel, (tSize, tSize)
                                tpatch = tslide.read_region((tl_l0_x, tl_l0_y), sstLevel, (tSize, tSize))
                                arr = tpatch
                                if arr.dtype != np.uint8:
                                    arr = arr.astype(np.uint8)
                                # 使用 PIL 处理通道/alpha 问题
                                tpatch_pil = Image.fromarray(arr)
                                if tpatch_pil.mode != 'RGB':
                                    tpatch_pil = tpatch_pil.convert('RGB')

                                # 信息含量过滤
                                if enable_information_filter:
                                    if not is_informative_patch(tpatch_pil, 
                                                              entropy_threshold=information_threshold,
                                                              std_threshold=min_std_threshold):
                                        # 如果信息含量不足，跳过这个patch
                                        continue
                                
                                saved_patches += 1

                                tname = f'{tslide_name}_{center_l0_for_target_x}_{center_l0_for_target_y}_{ix}_{iy}_WW_{grid_w}_HH_{grid_h}.{patch_surfix}'
                                tpatch_pil.save(os.path.join(tsave_folder, tname))
                            except Exception as ie:
                                num_error += 1
                                print(f'幻灯片 {tslide_name} 保存错误 {num_error}: {ie}')
        print(f'幻灯片 {tslide_name} 完成: 总共{total_patches}块, 保存{saved_patches}块, 错误{num_error}块')
    except Exception as e:
        print(f"提取patch时发生错误: {e}")
        raise

def getFolder_name(orig_dir, level, psize):
    tslide = os.path.basename(orig_dir)
    folderName = os.path.dirname(orig_dir)

    subfolder_name = float(psize * level) / 256
    tfolder = os.path.join(folderName, str(subfolder_name*10), tslide)
    return tfolder

def Thread_PatchFromSlides(args):
    normSlidePath, slideName, tsave_slide_dir = args
    
    print(f"开始处理: {slideName}")
    
    try:
        # 尝试打开幻灯片文件
        tslide = sdpc.Sdpc(normSlidePath)
        print(f"成功加载: {slideName}")
        
        # 检查幻灯片层级信息
        print(f"{slideName} 的层级数量: {tslide.level_count}")
        print(f"{slideName} 的层级尺寸: {tslide.level_dimensions}")
        
        # 自动调整mask_level
        actual_mask_level = min(mask_dimension_level, tslide.level_count - 1)
        if actual_mask_level != mask_dimension_level:
            print(f"{slideName}: 调整mask_level从{mask_dimension_level}到{actual_mask_level}")
        
        # 创建保存目录
        for tlevel, tsize in zip(patch_level_list, psize_list):
            tsave_dir_level = getFolder_name(tsave_slide_dir, tlevel, tsize)
            if not os.path.exists(tsave_dir_level):
                os.makedirs(tsave_dir_level)

        # 获取组织区域
        tissue_mask, boundingBoxes = get_roi_bounds(tslide, mask_level=actual_mask_level)
        
        # 修复：使用 np.isnan() 来检查 nan 值
        if tissue_mask is np.nan or np.isnan(tissue_mask).any():
            error_msg = f"无法生成组织掩模: {slideName}"
            print(error_msg)
            log_failed_slide(normSlidePath, error_msg)
            return

        tissue_mask = tissue_mask // 255

        # 提取patch
        Extract_Patch_From_Slide_STRIDE(tslide, tissue_mask, tsave_slide_dir,
                                      patch_level=patch_dimension_level, 
                                      mask_level=actual_mask_level,
                                      patch_stride=stride, 
                                      patch_size=psize,
                                      threshold=tissue_mask_threshold,
                                      level_list=patch_level_list,
                                      patch_size_list=psize_list)
        
    except Exception as e:
        error_msg = f"处理失败: {slideName} - {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        log_failed_slide(normSlidePath, error_msg)

    print(f"完成处理: {slideName}")


if __name__ == "__main__":
    # 清空之前的失败记录
    if os.path.exists(failed_slides_file):
        os.remove(failed_slides_file)
    
    arg_list = []
    
    for tSlidePath in slide_paths:
        slideName = os.path.basename(tSlidePath).split('.')[0]
        tsave_slide_dir = os.path.join(save_folder_dir, slideName)
        arg_list.append([tSlidePath, slideName, tsave_slide_dir])

    # 将arg_list按num_thread划分为子列表并存入一个列表
    arg_sublists = [arg_list[i:i + num_thread] for i in range(0, len(arg_list), num_thread)]
    
    print(f"准备处理 {len(arg_list)} 个幻灯片")
    
    # 使用进程池
    for sublist in arg_sublists:
        if not sublist:
            continue
        print(f"提交 {len(sublist)} 个任务到进程池")
        pool = Pool(processes=len(sublist))
        # 将子列表中的每个条目传入 Thread_PatchFromSlides（它会打开 slide 并调用 Extract_Patch_From_Slide_STRIDE）
        pool.map(Thread_PatchFromSlides, sublist)

    
    # 读取并显示失败记录
    if os.path.exists(failed_slides_file):
        with open(failed_slides_file, 'r') as f:
            failed_slides = f.readlines()
        print(f"\n处理完成！失败的幻灯片数量: {len(failed_slides)}")
        for line in failed_slides:
            print(line.strip())
    else:
        print("\n所有幻灯片处理成功！")