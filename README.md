# 排球動作識別專案 (Volleyball Action Recognition)

## 專案概述

本專案使用 YOLOv11m 模型進行排球動作識別，能夠檢測和分類五種不同的排球動作：攔網(block)、接球(receive)、發球(serve)、舉球(set)、扣球(spike)。

## 資料集資訊

### 資料集來源
本專案使用的資料集來自兩個公開的 Roboflow 資料集，並已整理為統一的格式：

**已整理資料集（推薦使用）**
- **Google Drive 下載**：[📁 排球動作識別資料集](https://drive.google.com/drive/folders/1lvWUwkBAEeCGJoM7Z5gwE71ngi94xQB5?usp=share_link)
- 包含完整的 train/valid/test 分割
- 已合併並統一標籤格式
- 包含 data.yaml 配置檔案

**原始資料集來源**
1. **Volleyball Actions Dataset**
   - 來源：https://universe.roboflow.com/actions-players/volleyball-actions/dataset/5
   - 工作空間：actions-players
   - 專案：volleyball-actions
   - 版本：5
   - 授權：CC BY 4.0

2. **Volleyball Action Recognition Dataset**
   - 來源：https://universe.roboflow.com/vbanalyzer/volleyball-action-recognition-k6tqv/dataset/6
   - 工作空間：vbanalyzer
   - 專案：volleyball-action-recognition
   - 版本：6

### 資料集統計
- **總圖片數量**：24,806 張
- **總標籤數量**：24,806 個
- **訓練集**：18,616 張圖片
- **驗證集**：3,636 張圖片
- **測試集**：2,554 張圖片

### 類別定義
| 類別ID | 類別名稱 | 英文名稱 | 描述 |
|--------|----------|----------|------|
| 0 | 攔網 | block | 球員在網前進行攔網動作 |
| 1 | 接球 | receive | 球員接發球或接扣球的動作 |
| 2 | 發球 | serve | 球員發球的動作 |
| 3 | 舉球 | set | 球員舉球給隊友的動作 |
| 4 | 扣球 | spike | 球員扣球攻擊的動作 |

## 技術規格

### 模型架構
- **模型**：YOLOv11m (Medium)
- **參數數量**：20,056,863
- **層數**：231 層
- **GFLOPs**：68.2

### 訓練配置
- **框架**：Ultralytics YOLO
- **設備**：Apple M1 Pro (MPS GPU加速) / NVIDIA RTX 5070 (CUDA GPU加速)
- **批次大小**：12 (M1 Pro) / 16-20 (RTX 5070)
- **圖像尺寸**：640x640
- **訓練輪數**：200 epochs
- **優化器**：SGD
- **學習率**：0.001
- **動量**：0.937
- **權重衰減**：0.0005
- **混合精度**：AMP (M1 Pro) / AMP + FP16 (RTX 5070)

### 資料增強
- **水平翻轉**：0.5
- **HSV調整**：色調±0.015，飽和度±0.7，明度±0.4
- **馬賽克增強**：1.0
- **混合增強**：0.0
- **複製貼上**：0.0

## 專案結構

```
vb_action_yolov11/
├── README.md                    # 專案說明文件
├── train_volleyball.py          # 主要訓練腳本
├── yolo11m.pt                   # YOLOv11m 預訓練模型
├── .gitignore                   # Git 忽略文件
├── venv/                        # Python虛擬環境 (不包含在版本控制中)
└── Volleyball_Action_Dataset/   # 合併後的資料集 (不包含在版本控制中)
    ├── data.yaml               # 資料集配置文件
    ├── train/                  # 訓練資料
    │   ├── images/            # 訓練圖片
    │   └── labels/            # 訓練標籤
    ├── valid/                  # 驗證資料
    │   ├── images/            # 驗證圖片
    │   └── labels/            # 驗證標籤
    └── test/                   # 測試資料
        ├── images/            # 測試圖片
        └── labels/            # 測試標籤
```

## 環境設定

### 系統需求
- **作業系統**：macOS (支援Apple Silicon) / Linux / Windows
- **Python版本**：3.8 或更高版本
- **記憶體**：建議16GB以上
- **儲存空間**：至少5GB可用空間
- **GPU**：建議使用 Apple Silicon (M1/M2) 或支援CUDA的GPU
  - **Apple Silicon**：M1 Pro/Max/Ultra 或 M2/M3 系列（主要支援）
  - **NVIDIA GPU**：RTX 5070 或更高（備選支援）
  - **CUDA版本**：11.8 或 12.1（NVIDIA GPU 需要）
  - **VRAM**：建議8GB以上（NVIDIA GPU）

### 安裝步驟

1. **克隆專案**
   ```bash
   git clone https://github.com/yourusername/vb_action_yolov11.git
   cd vb_action_yolov11
   ```

2. **建立虛擬環境**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # macOS/Linux
   # 或
   venv\Scripts\activate     # Windows
   ```

3. **安裝依賴套件**

   **方法一：使用 requirements.txt（推薦）**
   ```bash
   pip install -r requirements.txt
   ```

   **方法二：手動安裝**
   ```bash
   # 核心依賴
   pip install ultralytics
   pip install torch torchvision torchaudio
   pip install opencv-python pillow pyyaml
   
   # 視覺化工具
   pip install matplotlib seaborn tensorboard
   
   # 工具套件
   pip install tqdm psutil pandas
   ```

4. **設備特定配置**

   **Apple Silicon (M1/M2) 用戶**
   ```python
   # 驗證 MPS 支援
   import torch
   print(f"MPS 可用: {torch.backends.mps.is_available()}")
   print(f"MPS 建置: {torch.backends.mps.is_built()}")
   ```

   **NVIDIA GPU 用戶 (RTX 5070)**
   
   **檢查 CUDA 版本**
   ```bash
   nvidia-smi
   ```

   **安裝對應的 PyTorch CUDA 版本**
   ```bash
   # CUDA 11.8
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # CUDA 12.1
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

   **驗證 CUDA 安裝**
   ```python
   import torch
   print(f"CUDA 可用: {torch.cuda.is_available()}")
   print(f"CUDA 版本: {torch.version.cuda}")
   print(f"GPU 數量: {torch.cuda.device_count()}")
   print(f"GPU 名稱: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
   ```

5. **下載資料集**

   **方法一：使用已整理好的資料集（推薦）**
   - 從 Google Drive 下載已整理好的完整資料集：
   - [📁 排球動作識別資料集 - Google Drive](https://drive.google.com/drive/folders/1lvWUwkBAEeCGJoM7Z5gwE71ngi94xQB5?usp=share_link)
   - 下載後解壓縮並放置在 `Volleyball_Action_Dataset/` 目錄中
   - 資料集已包含完整的 train/valid/test 分割和 data.yaml 配置檔案

   **方法二：從原始來源下載**
   - 由於資料集檔案過大，也可以從以下原始連結下載：
   - [Volleyball Actions Dataset](https://universe.roboflow.com/actions-players/volleyball-actions/dataset/5)
   - [Volleyball Action Recognition Dataset](https://universe.roboflow.com/vbanalyzer/volleyball-action-recognition-k6tqv/dataset/6)
   - 將下載的資料集合併並放置在 `Volleyball_Action_Dataset/` 目錄中

## 使用方法

### 開始訓練
```bash
# 啟動虛擬環境
source venv/bin/activate  # macOS/Linux
# 或
venv\Scripts\activate     # Windows

# 開始訓練
python train_volleyball.py
```

### 訓練參數說明
- **epochs**: 200 - 訓練輪數
- **batch**: 12 - 批次大小 (M1 Pro) / 16-20 (RTX 5070)
- **imgsz**: 640 - 輸入圖像尺寸
- **device**: 'mps' - 使用 Apple Silicon GPU (M1 Pro) / 'cuda' (RTX 5070)
- **patience**: 50 - 早停耐心值
- **save_period**: 20 - 每20個epoch保存檢查點
- **amp**: True - 自動混合精度訓練
- **half**: False - 半精度 (M1 Pro 不支援) / True (RTX 5070 支援)
- **workers**: 4 - 資料載入器工作進程數 (M1 Pro) / 8-12 (RTX 5070)

### 訓練輸出
訓練過程中會顯示以下指標：
- **mAP50**：平均精度 (IoU=0.5)
- **mAP50-95**：平均精度 (IoU=0.5:0.95)
- **Precision**：精確度
- **Recall**：召回率
- **F1 Score**：F1分數
- **Box Loss**：邊界框損失
- **Class Loss**：分類損失
- **DFL Loss**：DFL損失

### 檢查點保存
- 每20個epoch自動保存檢查點
- 最佳模型：`runs/volleyball_200epoch/weights/best.pt`
- 最新模型：`runs/volleyball_200epoch/weights/last.pt`

### 模型評估
訓練完成後，腳本會自動在測試集上評估模型性能。

## 使用預訓練模型進行推理

```python
from ultralytics import YOLO

# 載入訓練好的模型
model = YOLO('runs/volleyball_200epoch/weights/best.pt')

# 對單張圖片進行預測
results = model('path/to/image.jpg')

# 顯示結果
results[0].show()
```

## 資料集授權

本專案使用的資料集遵循以下授權：
- **CC BY 4.0**：允許商業使用、修改、分發，但需標明原作者
- 詳細授權條款請參考：https://creativecommons.org/licenses/by/4.0/

## 引用

如果您在研究中使用了本專案，請引用相關的資料集：

```bibtex
@dataset{volleyball_actions_2023,
  title={Volleyball Actions Dataset},
  author={actions-players},
  year={2023},
  url={https://universe.roboflow.com/actions-players/volleyball-actions/dataset/5},
  license={CC BY 4.0}
}

@dataset{volleyball_action_recognition_2023,
  title={Volleyball Action Recognition Dataset},
  author={vbanalyzer},
  year={2023},
  url={https://universe.roboflow.com/vbanalyzer/volleyball-action-recognition-k6tqv/dataset/6}
}
```

## 常見問題

### Q: 如何調整訓練參數？
A: 修改 `train_volleyball.py` 中的 `training_args` 字典。

### Q: 支援哪些設備？
A: 主要支援 Apple Silicon (M1/M2)，也支援 CUDA GPU、CPU 訓練。

### Q: 如何為不同設備調整設定？
A: 
**Apple Silicon (M1 Pro) 推薦設定**:
- **批次大小**: 12
- **設備**: 'mps'
- **混合精度**: AMP (half=False)
- **工作進程**: 4

**RTX 5070 推薦設定**:
- **批次大小**: 16-20 (根據 VRAM 調整)
- **設備**: 'cuda'
- **混合精度**: AMP + FP16 (half=True)
- **工作進程**: 8-12
- **CUDA 版本**: 11.8 或 12.1
- **VRAM 使用**: 約 6-8GB (批次大小 16)

### Q: 如何從檢查點繼續訓練？
A: 將檢查點檔案放在 `runs/volleyball_200epoch/weights/` 目錄中，腳本會自動檢測並繼續訓練。

### Q: 記憶體不足怎麼辦？
A: 可以降低批次大小 (`batch`) 或關閉圖片緩存 (`cache: False`)。

### Q: 如何下載資料集？
A: 推薦從 [Google Drive](https://drive.google.com/drive/folders/1lvWUwkBAEeCGJoM7Z5gwE71ngi94xQB5?usp=share_link) 下載已整理好的資料集，包含完整的 train/valid/test 分割和配置檔案。

## 聯絡資訊

如有任何問題或建議，請聯繫專案維護者。

---

**注意**：本專案僅供學術研究使用，請遵守相關資料集的授權條款。