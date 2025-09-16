#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
排球動作檢測訓練腳本 - 200個epoch版本
使用 YOLOv11m 模型訓練排球動作識別，從檢查點繼續訓練
"""

from ultralytics import YOLO
import os
import yaml

def main():
    # 設定資料集路徑
    data_yaml_path = "Volleyball_Action_Dataset/data.yaml"
    
    # 檢查資料集配置檔案是否存在
    if not os.path.exists(data_yaml_path):
        print(f"錯誤：找不到資料集配置檔案 {data_yaml_path}")
        return
    
    # 讀取並顯示資料集資訊
    with open(data_yaml_path, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)
    
    print("=== 資料集資訊 ===")
    print(f"類別數量: {data_config['nc']}")
    print(f"類別名稱: {data_config['names']}")
    print(f"訓練集路徑: {data_config['train']}")
    print(f"驗證集路徑: {data_config['val']}")
    print(f"測試集路徑: {data_config['test']}")
    print("==================")
    
    # 檢查是否有現有的檢查點
    checkpoint_path = "runs/volleyball_200epoch/weights/last.pt"
    if os.path.exists(checkpoint_path):
        print(f"找到檢查點: {checkpoint_path}")
        print("將從檢查點繼續訓練...")
        model = YOLO(checkpoint_path)
    else:
        print("未找到檢查點，從預訓練模型開始...")
        model = YOLO('yolo11m.pt')
    
    # 設定訓練參數 - 優化版本
    training_args = {
        'data': data_yaml_path,
        'epochs': 200,  # 訓練輪數 (200個epochs)
        'imgsz': 640,   # 輸入圖像大小
        'batch': 12,    # 批次大小 (M1 Pro優化)
        'device': 'mps',  # 使用 M1 GPU 訓練 (Metal Performance Shaders)
        'project': 'runs',  # 專案目錄
        'name': 'volleyball_200epoch',  # 實驗名稱
        'save': True,   # 儲存檢查點
        'save_period': 20,  # 每20個epoch儲存一次
        'cache': False,  # 關閉圖片緩存 (避免記憶體不足)
        'workers': 4,   # 資料載入器工作進程數 (M1 Pro優化)
        'patience': 50,  # 早停耐心值 (增加耐心值)
        'lr0': 0.001,   # 初始學習率 (降低學習率)
        'lrf': 0.1,     # 最終學習率
        'momentum': 0.937,  # 動量
        'weight_decay': 0.0005,  # 權重衰減
        'warmup_epochs': 3,  # 預熱輪數
        'warmup_momentum': 0.8,  # 預熱動量
        'warmup_bias_lr': 0.1,   # 預熱偏置學習率
        'box': 7.5,     # 邊界框損失權重
        'cls': 0.5,     # 分類損失權重
        'dfl': 1.5,     # DFL損失權重
        'pose': 12.0,   # 姿態損失權重
        'kobj': 2.0,    # 關鍵點物件損失權重
        'nbs': 64,      # 標稱批次大小
        'overlap_mask': True,  # 訓練時重疊遮罩
        'mask_ratio': 4,  # 遮罩下採樣比例
        'dropout': 0.0,  # Dropout率
        'val': True,    # 訓練期間驗證
        'plots': False,  # 暫時關閉繪圖 (避免錯誤)
        'verbose': True,  # 詳細輸出
        'resume': False,  # 不從檢查點繼續訓練（因為沒有檢查點）
        'save_json': True,  # 保存JSON格式結果
        'save_txt': True,   # 保存檢測結果文本
        'save_conf': True,  # 保存置信度分數
        'amp': True,    # 自動混合精度 (M1 Pro優化)
        'half': False,  # 不使用半精度 (M1 Pro不支援)
        'dnn': False,   # 不使用OpenCV DNN
        'deterministic': True,  # 確定性訓練
    }
    
    print("開始訓練...")
    print(f"訓練參數: {training_args}")
    
    # 開始訓練
    try:
        results = model.train(**training_args)
        print("訓練完成！")
        print(f"最佳模型儲存在: runs/volleyball_200epoch/weights/best.pt")
        print(f"最後模型儲存在: runs/volleyball_200epoch/weights/last.pt")
        
        # 顯示訓練結果摘要
        print("\n=== 訓練結果摘要 ===")
        print("🎯 主要指標:")
        print(f"  mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")
        print(f"  mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
        print(f"  Precision: {results.results_dict.get('metrics/precision(B)', 'N/A'):.4f}")
        print(f"  Recall: {results.results_dict.get('metrics/recall(B)', 'N/A'):.4f}")
        print(f"  F1 Score: {results.results_dict.get('metrics/f1', 'N/A'):.4f}")
        
        print("\n📊 損失函數:")
        print(f"  Box Loss: {results.results_dict.get('train/box_loss', 'N/A'):.4f}")
        print(f"  Class Loss: {results.results_dict.get('train/cls_loss', 'N/A'):.4f}")
        print(f"  DFL Loss: {results.results_dict.get('train/dfl_loss', 'N/A'):.4f}")
        
        print("\n🔧 訓練配置:")
        print(f"  總Epochs: {training_args['epochs']}")
        print(f"  批次大小: {training_args['batch']}")
        print(f"  設備: {training_args['device']}")
        print(f"  圖像尺寸: {training_args['imgsz']}")
        
        print("\n💾 模型保存位置:")
        print(f"  最佳模型: runs/volleyball_200epoch/weights/best.pt")
        print(f"  最新模型: runs/volleyball_200epoch/weights/last.pt")
        
    except Exception as e:
        print(f"訓練過程中發生錯誤: {e}")
        return
    
    # 在測試集上評估模型
    print("\n在測試集上評估模型...")
    try:
        test_results = model.val(data=data_yaml_path, split='test')
        print("測試集評估完成！")
    except Exception as e:
        print(f"測試集評估時發生錯誤: {e}")

if __name__ == "__main__":
    main()
