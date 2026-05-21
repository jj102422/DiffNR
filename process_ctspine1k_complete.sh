#!/bin/bash

################################################################################
# 完整 CTSpine1K 数据处理流程
# 步骤 1: 生成投影和元数据
# 步骤 2: 初始化高斯点云
# 步骤 3: 训练高斯模型并生成预测体积
################################################################################

set -e  # 任何错误立即退出

export CUDA_VISIBLE_DEVICES=0
cd /home/jym/DiffNR

DATA_ROOT="/home/public/CTSpine1K/data/data-MHD_ctpro_woMask1"
SCANNER_CFG="r2_gaussian/data_generator/scanner/cone_beam.yml"
TEMP_OUTPUT_ROOT="/tmp/diffnr_outputs"  # 临时输出目录
CONFIG="configs/luna16.yaml"

# ============ 训练参数（同步自 run_train_DiffNR.sh） ============
ORGAN_TYPE="colon"  # 默认器官类型
LAMBDA_DIFFUSION_SSIM="0"  # 扩散损失权重（默认不使用）

# ============ 配置文件参数（来自 configs/luna16.yaml） ============
ITERATIONS="12000"  # 训练迭代次数（luna16.yaml 默认值）
LAMBDA_DSSIM="0.25"  # 投影空间 DSSIM 损失权重
LAMBDA_TV="0.05"  # 体积正则化权重
TV_VOL_SIZE="32"  # TV loss 体积大小
LAMBDA_DIFFUSION_L1="0.0"  # 扩散 L1 损失权重
DIFFUSION_TV_SIZE="0"  # 扩散 TV 体积大小
DENSIFICATION_INTERVAL="100"  # 密化间隔
DENSIFY_FROM_ITER="500"  # 从第几步开始密化
DENSIFY_UNTIL_ITER="0"  # 密化到第几步（0表示不限制）
DENSIFY_GRAD_THRESHOLD="0.00005"  # 密化梯度阈值

# ============ 初始化参数（来自 train_all_save_to_case.py） ============
INIT_RECON_METHOD="fdk"  # 初始化重建方法
INIT_N_POINTS="50000"  # 初始化点云数量
INIT_DENSITY_THRESH="0.05"  # 初始化密度阈值
INIT_DENSITY_RESCALE="0.15"  # 初始化密度缩放因子
INIT_RANDOM_DENSITY_MAX="1.0"  # 随机初始化最大密度

# ============ 扫描仪配置（来自 scanner/cone_beam.yml） ============
# Mode: cone（圆锥束）
# DSD: 6.0（源到探测器距离）
# DSO: 4.0（源到原点距离）
# nDetector: [256, 256]（探测器像素数）
# nVoxel: [512, 512, ...]（体素数）
# sDetector: [3.0, 3.0]（探测器大小）

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}===============================================${NC}"
echo -e "${GREEN}CTSpine1K 完整数据处理流程${NC}"
echo -e "${GREEN}===============================================${NC}"
echo ""
echo "配置信息:"
echo "  Python 解释器: /home/jym/python"
echo "  GPU 卡号: 0 (CUDA_VISIBLE_DEVICES=0)"
echo "  工作目录: /home/jym/DiffNR"
echo ""
echo "训练参数（同步自 run_train_DiffNR.sh）:"
echo "  器官类型 (--organ_type): $ORGAN_TYPE"
echo "  扩散损失权重 (--lambda_diffusion_ssim): $LAMBDA_DIFFUSION_SSIM"
echo ""
echo "配置文件参数（来自 configs/luna16.yaml）:"
echo "  迭代次数 (iterations): $ITERATIONS"
echo "  投影 DSSIM 损失 (lambda_dssim): $LAMBDA_DSSIM"
echo "  体积 TV 损失 (lambda_tv): $LAMBDA_TV"
echo "  TV 体积大小 (tv_vol_size): $TV_VOL_SIZE"
echo "  扩散 L1 损失 (lambda_diffusion_l1): $LAMBDA_DIFFUSION_L1"
echo "  扩散 TV 大小 (diffusion_tv_size): $DIFFUSION_TV_SIZE"
echo "  密化间隔 (densification_interval): $DENSIFICATION_INTERVAL"
echo "  密化起始步 (densify_from_iter): $DENSIFY_FROM_ITER"
echo ""
echo "初始化参数（来自 train_all_save_to_case.py）:"
echo "  重建方法 (--init_recon_method): $INIT_RECON_METHOD"
echo "  初始化点数 (--init_n_points): $INIT_N_POINTS"
echo "  密度阈值 (--init_density_thresh): $INIT_DENSITY_THRESH"
echo "  密度缩放 (--init_density_rescale): $INIT_DENSITY_RESCALE"
echo ""
echo "扫描仪参数（来自 scanner/cone_beam.yml）:"
echo "  模式: cone（圆锥束）"
echo "  源到探测器距离 (DSD): 6.0"
echo "  源到原点距离 (DSO): 4.0"
echo "  探测器像素数 (nDetector): [256, 256]"
echo "  探测器大小 (sDetector): [3.0, 3.0]"
echo "  体素数量 (nVoxel): [512, 512, ...]"
echo ""

TOTAL_CASES=$(find "$DATA_ROOT" -mindepth 1 -maxdepth 1 -type d | awk 'END{print NR}')
VOLUME_GT_COUNT=$(find "$DATA_ROOT" -mindepth 2 -maxdepth 2 -name volume_gt.npy | awk 'END{print NR}')
INIT_COUNT=$(find "$DATA_ROOT" -mindepth 2 -maxdepth 2 -name 'init_*.npy' | awk 'END{print NR}')

################################################################################
# 第一步：生成所有数据（投影、元数据、X-ray特征）
################################################################################
echo -e "${YELLOW}[步骤 1] 生成投影和元数据（约 10-15 分钟）${NC}"
echo "命令: generate_data.py --data_root $DATA_ROOT --scanner $SCANNER_CFG"
echo ""
if [ "$TOTAL_CASES" -gt 0 ] && [ "$VOLUME_GT_COUNT" -ge "$TOTAL_CASES" ]; then
    echo -e "${GREEN}✅ 步骤 1 已完成，跳过。${NC}"
    echo ""
else
    /home/jym/python r2_gaussian/data_generator/synthetic_dataset/generate_data.py \
      --data_root "$DATA_ROOT" \
      --scanner "$SCANNER_CFG" \
      --ct_name ct_file.mha

    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ 步骤 1 失败！生成数据出错。${NC}"
        exit 1
    fi

    echo -e "${GREEN}✅ 步骤 1 完成！${NC}"
    echo ""
fi

################################################################################
# 第二步：初始化所有 case 的高斯点云
################################################################################
echo -e "${YELLOW}[步骤 2] 初始化高斯点云（约 3-5 分钟/case）${NC}"
echo ""

if [ "$TOTAL_CASES" -gt 0 ] && [ "$INIT_COUNT" -ge "$TOTAL_CASES" ]; then
    echo -e "${GREEN}✅ 步骤 2 已完成，跳过。${NC}"
    echo ""
else
    CASE_COUNT=0
    for case_dir in "$DATA_ROOT"/*/; do
        if [ ! -d "$case_dir" ]; then
            continue
        fi
        
        case_id=$(basename "$case_dir")
        init_path="${case_dir}init_${case_id}.npy"
        
        echo "处理: $case_id"
        
        # 检查必要的输入文件
        if [ ! -f "${case_dir}meta_data.json" ] || [ ! -f "${case_dir}volume_gt.npy" ]; then
            echo -e "${RED}  ⚠️  跳过 $case_id: 缺少 meta_data.json 或 volume_gt.npy${NC}"
            continue
        fi
        
        # 已经存在则直接跳过，避免重复初始化
        if [ -s "$init_path" ]; then
            echo -e "${GREEN}  ✅ $case_id 已存在初始化文件，跳过。${NC}"
            echo ""
            CASE_COUNT=$((CASE_COUNT + 1))
            continue
        fi
        
        # 运行初始化
        /home/jym/python r2_gaussian/data_generator/initialize_pcd.py \
          --data "$case_dir" \
          --output "$init_path" \
          --recon_method fdk \
          --n_points 50000 \
          --density_thresh 0.05 \
          --density_rescale 0.15
        
        if [ $? -ne 0 ]; then
            echo -e "${RED}  ⚠️  $case_id 初始化失败，继续下一个...${NC}"
            continue
        fi
        
        echo -e "${GREEN}  ✅ $case_id 初始化完成！${NC}"
        echo ""
        CASE_COUNT=$((CASE_COUNT + 1))
    done

    if [ $CASE_COUNT -eq 0 ]; then
        echo -e "${RED}❌ 步骤 2 失败！没有成功初始化任何 case。${NC}"
        exit 1
    fi

    echo -e "${GREEN}✅ 步骤 2 完成！共初始化 $CASE_COUNT 个 case。${NC}"
    echo ""
fi

################################################################################
# 第三步：训练高斯模型并生成预测体积
################################################################################
echo -e "${YELLOW}[步骤 3] 训练高斯模型（约 30-60 分钟/case，取决于迭代次数）${NC}"
echo "临时输出目录: $TEMP_OUTPUT_ROOT"
echo "最终文件将保存到: $DATA_ROOT/{case_id}/"
echo ""

# 创建临时输出目录
mkdir -p "$TEMP_OUTPUT_ROOT"

# 如果 volume_gt.npy 比历史预测结果更新，则删除旧结果触发步骤 3 重建
REBUILD_STALE_RESULTS="true"
STALE_CASES=0
if [ "$REBUILD_STALE_RESULTS" = "true" ]; then
    echo "检查是否存在需要重建的旧结果..."
    for case_dir in "$DATA_ROOT"/*/; do
        if [ ! -d "$case_dir" ]; then
            continue
        fi

        case_id=$(basename "$case_dir")
        volume_gt_path="${case_dir}volume_gt.npy"
        vol_pred_path="${case_dir}vol_pred.npy"
        point_cloud_path="${case_dir}point_cloud.pickle"

        if [ ! -f "$volume_gt_path" ]; then
            continue
        fi

        needs_rebuild="0"
        if [ ! -f "$vol_pred_path" ] || [ ! -f "$point_cloud_path" ]; then
            needs_rebuild="1"
        elif [ "$volume_gt_path" -nt "$vol_pred_path" ] || [ "$volume_gt_path" -nt "$point_cloud_path" ]; then
            needs_rebuild="1"
        fi

        if [ "$needs_rebuild" = "1" ]; then
            rm -f "$vol_pred_path" "$point_cloud_path"
            echo "  标记重建: $case_id"
            STALE_CASES=$((STALE_CASES + 1))
        fi
    done
    echo "需要重建的 case 数量: $STALE_CASES"
    echo ""
fi

# 检查是否有 SliceFixer 模型（可选）
SLICEFIXER_CKPT=""
if [ -f "/home/public/CTSpine1K/data/diffnr/slicefixer_model.pkl" ]; then
    SLICEFIXER_CKPT="/home/public/CTSpine1K/data/diffnr/slicefixer_model.pkl"
    echo "检测到 SliceFixer 模型: $SLICEFIXER_CKPT"
fi

# 检查 SD-Turbo 路径（可选）
SD_TURBO_PATH=""
if [ -d "/home/public/CTSpine1K/data/diffnr/sd-turbo" ]; then
    SD_TURBO_PATH="/home/public/CTSpine1K/data/diffnr/sd-turbo"
    echo "检测到 SD-Turbo 模型: $SD_TURBO_PATH"
fi

echo ""

# 运行训练
if [ -n "$SLICEFIXER_CKPT" ] && [ -n "$SD_TURBO_PATH" ]; then
    /home/jym/python scripts/train_all_save_to_case.py \
      --source "$DATA_ROOT" \
      --output "$TEMP_OUTPUT_ROOT" \
      --config "$CONFIG" \
      --device 0 \
      --ckpt "$SLICEFIXER_CKPT" \
      --sd_turbo_path "$SD_TURBO_PATH" \
      --organ_type "$ORGAN_TYPE" \
            --skip_existing
else
    # 不使用 SliceFixer
    /home/jym/python scripts/train_all_save_to_case.py \
      --source "$DATA_ROOT" \
      --output "$TEMP_OUTPUT_ROOT" \
      --config "$CONFIG" \
      --device 0 \
      --organ_type "$ORGAN_TYPE" \
            --skip_existing
fi

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ 步骤 3 失败！训练出错。${NC}"
    exit 1
fi

echo -e "${GREEN}✅ 步骤 3 完成！${NC}"
echo ""

################################################################################
# 完成
################################################################################
echo -e "${GREEN}===============================================${NC}"
echo -e "${GREEN}✅ 所有步骤完成！${NC}"
echo -e "${GREEN}===============================================${NC}"
echo ""
echo "数据位置:"
echo "  源数据: $DATA_ROOT"
echo "  临时输出: $TEMP_OUTPUT_ROOT"
echo ""
echo "最终生成的文件位置（在每个 case 目录）："
echo "  ✅ volume_gt.npy (归一化的 CT 体积 [0,1])"
echo "  ✅ proj_train/*.npy (训练投影)"
echo "  ✅ proj_test/*.npy (测试投影)"
echo "  ✅ {case}_xray_1.pt, {case}_xray_2.pt (X-ray 特征)"
echo "  ✅ meta_data.json (扫描参数)"
echo "  ✅ init_{case}.npy (初始化点云)"
echo "  ✅ vol_pred.npy (预测的 CT 体积 [0,1]) ← 在 case 目录"
echo "  ✅ point_cloud.pickle (训练后的高斯参数) ← 在 case 目录"
echo ""
echo "示例路径:"
echo "  $DATA_ROOT/volume-covid19-A-0237_ct/"
echo "    ├── volume_gt.npy"
echo "    ├── vol_pred.npy ✨"
echo "    ├── point_cloud.pickle ✨"
echo "    ├── init_volume-covid19-A-0237_ct.npy"
echo "    ├── meta_data.json"
echo "    ├── proj_train/"
echo "    ├── proj_test/"
echo "    ├── volume-covid19-A-0237_ct_xray_1.pt"
echo "    └── volume-covid19-A-0237_ct_xray_2.pt"
echo ""
