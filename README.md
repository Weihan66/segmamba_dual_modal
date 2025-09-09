# segmamba_dual_modal

# SegMamba Dual-Modal for 3D Breast Tumor Segmentation

结构：CT 分支 (SegMamba-like) + PET 分支 (3D CNN) + CMA 融合 (SwinCross-inspired).

## 运行
1. 准备数据：将配对的 CT/PET NIfTI 放到 `./data/train/` 和 `./data/val/`，每个病例放成两文件：
   - `<case_id>_CT.nii.gz`
   - `<case_id>_PET.nii.gz`
   - 标注：`<case_id>_GT.nii.gz`

2. 安装依赖（示例）：
   pip install torch torchvision numpy nibabel scikit-image tqdm
   optional: pip install monai

3. 修改 `configs.py` 中路径与超参。

4. 训练：
   python train.py

5. 验证/推理：
   python evaluate.py --ckpt checkpoints/best.pth

## 注意
- 代码为可运行参考实现；可替换任意模块（替换为作者原实现）以进一步提升性能。
