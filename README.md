# 眼底血管分割（Retinal Vessel Segmentation）

基于 U-Net 的 2D 眼底图像视网膜血管分割，配合外部预训练模型的互补融合，在竞赛中取得 **Dice 0.78**。

## 任务

从彩色眼底图像中分割出视网膜血管区域，按竞赛规定的 Run-Length Encoding（RLE）格式提交。
训练集 20 张（565×584，含标注），测试集 20 张，评价指标为 Dice 系数。

## 结果

| 方法配置 | 交叉验证 Dice | 提交 Dice |
|---|---|---|
| ResNet34 + 5 折，固定阈值 | 0.7770 | 0.74 |
| EfficientNet-B4 + 5 折 + 24 路 TTA | 0.7770 | 0.76 |
| ResNet34 + 5 折 + FOV 后处理 | 0.7696 | 0.76 |
| **ResNet34 + 5 折 + VascX 并集融合** | **0.7844** | **0.78** |

## 关键发现

1. **提交格式的语义陷阱**：官方 `segmentation_to_csv.py` 把所有「大于 0」的像素判为血管。
   因此预测图必须是 **血管=255、背景=0**，且尺寸必须是 **512×512**（脚本内部会重采样，
   尺寸不符会导致边界插值产生大量假阳性）。实测 400×400 输入会使编码像素数从 23094 膨胀到 35370。

2. **FOV 掩膜后处理无效**：实测增益精确为 0.0000。模型在训练中已从标注学到「视野外即背景」，
   视野外假阳性平均每张仅 1.4 像素。

3. **异源模型误差互补**：VascX 单独使用仅 0.7259（比本方法低 4.4 个点），因其偏保守、漏检细血管；
   但与本地模型取并集后达到 0.7844，超过任一单模型。交集则只有 0.7043。

## 目录结构

```
src/
  train_local.py              训练 + 推理 + 提交生成（5 折、OOF、TTA），支持断点续训
  vascx_batch.py              VascX 逐张推理（规避内存峰值）
  make_union_submission.py    并集融合与最终提交
notebooks/
  vessel_segmentation_kaggle.ipynb   Kaggle GPU 版本
report/
  build_report4.py            实验报告生成脚本
  figures/                    报告配图
run/                          纯 U-Net 的提交结果
union_run/                    融合后的最终提交结果
```

## 环境

```
torch >= 2.0
torchvision
segmentation-models-pytorch
albumentations
opencv-python
numpy
pandas
scikit-learn
```

CPU 即可运行。单折 80 轮约 12 分钟，5 折合计约 65 分钟。

## 复现

```bash
pip install -r requirements.txt

# 1. 训练 + 推理 + 生成提交
python src/train_local.py --folds 5 --epochs 80 --size 512 --out run

# 2. 融合外部模型结果（可选）
python src/make_union_submission.py

# 3. 生成实验报告
python report/build_report4.py
```

数据不包含在仓库中，需按 `data/segmentation/{train,test}` 结构放置。

> 国内网络下 HuggingFace 不可达，脚本已默认设置 `HF_ENDPOINT=https://hf-mirror.com`。

## 可复现性

固定了 Python / NumPy / PyTorch / CUDA 的全部随机种子，设置 cuDNN 为确定性模式，
并为每个 DataLoader 工作进程单独播种。同一硬件与依赖版本下结果可复现。

## 说明

课程作业代码，仅供学习参考。模型结构来自开源库
[segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch)，
预处理、训练策略、融合与后处理为本实验自主实现。
