# FirstTry

基于 Adult 数据集的成员推断攻击（Membership Inference Attack, MIA）实验代码，包含目标模型（非隐私/DP）训练、影子模型训练、攻击数据构建与评估流程。

## 目录结构

- Code/：主要代码与训练、评估脚本
- AdultsData/：Adult 数据集（adult.csv 等）
- results/：攻击数据与评估图表输出
- models/：历史模型权重（可选）

## 环境准备

```bash
pip install -r requirements.txt
```

## 本地环境

- torch 2.10.0+cu128
- CUDA 12.8
- GPU NVIDIA GeForce RTX 5080 Laptop GPU

## 快速运行

1. 预处理与数据切分（生成 Code/checkpoints 下的张量）

```bash
python -c "from Code.data_utils import prepare_and_save; prepare_and_save()"
```

2. 训练影子模型

```bash
python -c "from Code.train_shadow import train_shadow; train_shadow()"
```

3. 训练目标模型（非隐私 + DP）

```bash
python Code/train_target.py
```

4. 构建攻击数据并训练攻击模型

```bash
python Code/attack_analysis.py
```

5. 评估攻击效果（输出 ROC 与 ASR 等指标）

```bash
python Code/evaluation_pipeline.py
```

## 输出说明

- Code/checkpoints/：训练过程中的数据切分与模型权重
- results/attack_dataset.npz：攻击训练数据集
- results/roc_np.png / results/roc_dp.png：ROC 曲线图

## 参数说明

- 数据预处理默认读取 AdultsData/adult.csv，默认只取前 20 行，可在 prepare_and_save 中调整 n_rows
- 训练默认使用 GPU（可用时），否则自动回退到 CPU
