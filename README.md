# 多层感知机（MLP）——Fashion‑MNIST 分类示例

> 使用 **PyTorch** 实现的最小多层感知机（784‑256‑10），在 **Fashion‑MNIST** 数据集上训练 20 个 epoch，可在 **CPU/GPU** 上运行。

---

## 1️⃣ 项目亮点

| 模块 | 说明 |
|------|------|
| **极简网络** | `Flatten → Linear(784→256) → ReLU → Linear(256→10)` |
| **权重初始化** | `nn.init.normal_`，`std=0.01`；可改为 Xavier / Kaiming |
| **批训练** | `batch_size = 256`，在 GTX‑1650 上 20 个 epoch < 1 分钟 |
| **结果对比** | 最高 **测试准确率 ≈ 87%**，符合 MLP 在该数据集的预期 |

---

## 2️⃣ 环境要求

```text
Python  ≥ 3.8
PyTorch ≥ 2.0
TorchVision ≥ 0.15
```

建议创建虚拟环境：

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install torch torchvision
```

---

## 3️⃣ 快速开始

```bash
# 克隆仓库
git clone https://github.com/XiaoyuYang001/mlp-notebook.git
cd mlp-notebook

# 运行 notebook 或 python 脚本
jupyter notebook 多层感知机的实现.ipynb
# 或
python mlp_train.py
```

脚本参数（可选）：

```bash
python mlp_train.py \
  --batch_size 128 \
  --lr 0.05 \
  --epochs 30
```

---

## 4️⃣ 训练日志（默认超参，GPU）

```text
Epoch 1  | Loss 0.978 | Train 0.656 | Test 0.741
Epoch 10 | Loss 0.359 | Train 0.872 | Test 0.857
Epoch 20 | Loss 0.299 | Train 0.893 | Test 0.843
```

- **训练集准确率** 最终 ~89 %
- **测试集准确率** 最终 ~87 %

> MLP 无卷积，性能已接近上限。若需进一步提升，可使用 CNN（LeNet/ResNet）。

---

## 5️⃣ 目录结构

```
mlp-notebook/
├─ 多层感知机的实现.ipynb      # 主 notebook
├─ mlp_train.py                # 同等逻辑的脚本版（可选）
├─ requirements.txt            # 依赖列表（可选）
└─ README.md                   # 你正在看的文件
```

---

## 6️⃣ TODO

- [ ] 使用 **AdamW + CosineLR** 再提升 1~2 %
- [ ] 尝试 **BatchNorm / Dropout** 抑制过拟合
- [ ] 迁移到 **CNN** 结构突破 90 %

欢迎 PR & Issue ！

---

## 7️⃣ License

项目采用 **MIT License** ；代码与笔记自由复制、修改与分发，但请保留署名。

