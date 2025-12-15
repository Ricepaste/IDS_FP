import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
)
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# (省略 log_data_generator.py 的生成代碼, 假設 log_file 已存在)
LOG_FILE = "apache_log_dataset.log"


# 使用一個簡化的 Log Parser 來讀取 log_file
def load_and_parse_logs(log_file):
    """讀取日誌檔並解析為 DataFrame"""
    print("Parsing log file...")
    data = []
    log_pattern = re.compile(
        r"^(?P<ip>\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}) - - \[(?P<time>.*?)\] "
        r'"(?P<request>.*?)" (?P<status>\d+) (?P<bytes>\d+) '
        r'"(?P<referer>.*?)" "(?P<user_agent>.*?)" \| (?P<label>.*)'
    )

    with open(log_file, "r") as f:
        for line in tqdm(f, desc="Reading Logs"):
            match = log_pattern.match(line)
            if match:
                d = match.groupdict()
                # 從 request 中分離出 method 和 uri
                parts = d["request"].split(" ")
                d["method"] = parts[0] if len(parts) > 0 else "UNKNOWN"
                d["uri"] = parts[1] if len(parts) > 1 else "UNKNOWN"
                data.append(d)

    df = pd.DataFrame(data)
    # 將標籤數值化 (Normal=0, Anomaly=1)
    df["is_anomaly"] = df["label"].apply(lambda x: 0 if x == "NORMAL" else 1)
    df["status"] = pd.to_numeric(df["status"])
    return df


def feature_engineer(df: pd.DataFrame):
    """對 DataFrame 進行特徵工程"""

    # 1. 數值特徵: Bytes Sent, Status Code
    df["bytes"] = pd.to_numeric(df["bytes"])

    # 2. 類別特徵: Method
    method_encoded = pd.get_dummies(df["method"], prefix="method")

    # 3. Request URI 特徵 (簡單 Token 數量)
    df["uri_len"] = df["uri"].apply(lambda x: len(x.split("/")))

    # 4. IP 編碼 (將 IP 視為類別特徵，但只取最常見的 IP，其他設為 UNKNOWN)
    top_ips = df["ip"].value_counts().head(20).index
    df["ip_cat"] = df["ip"].apply(lambda x: x if x in top_ips else "OTHER_IP")
    ip_encoded = pd.get_dummies(df["ip_cat"], prefix="ip")

    # 5. User-Agent 類別 (簡單區分常見瀏覽器/工具)
    def categorize_ua(ua):
        if "Mozilla" in ua and "Chrome" in ua:
            return "Chrome"
        if "Mozilla" in ua and "Safari" in ua and "Chrome" not in ua:
            return "Safari"
        if "python-requests" in ua or "curl" in ua:
            return "Tool"
        if any(tool in ua for tool in ["sqlmap", "Nikto", "masscan"]):
            return "Malicious_Tool"
        return "Other_UA"

    df["ua_cat"] = df["user_agent"].apply(categorize_ua)
    ua_encoded = pd.get_dummies(df["ua_cat"], prefix="ua")

    # 合併所有特徵
    features_df = pd.concat(
        [df[["bytes", "status", "uri_len"]], method_encoded, ip_encoded, ua_encoded],
        axis=1,
    )

    # 數值標準化
    scaler = StandardScaler()
    # 選擇所有數值欄位（包括 one-hot 編碼後的）
    features_scaled = scaler.fit_transform(features_df)

    return features_scaled, scaler, features_df.columns


# 創建 PyTorch Dataset
class LogDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


import torch.nn as nn
import torch.optim as optim


class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=32):
        super(Autoencoder, self).__init__()
        # 編碼器 (Encoder)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim * 2),
            nn.ReLU(True),
            nn.Linear(encoding_dim * 2, encoding_dim),
            nn.ReLU(True),
        )

        # 解碼器 (Decoder)
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, encoding_dim * 2),
            nn.ReLU(True),
            nn.Linear(encoding_dim * 2, input_dim),
            # 使用 Sigmoid 或 Tanh 取決於數據分佈，但對於標準化數據，通常不需要
            # nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def train_model(model, train_loader, epochs, device):
    """使用正常資料訓練 Autoencoder"""
    criterion = nn.MSELoss()  # 重建誤差使用均方誤差
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    print(f"\nTraining Autoencoder on {device}...")
    for epoch in range(epochs):
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    print("Training finished.")


def predict_anomalies(model, data_loader, device, threshold):
    """在整個資料集上進行預測，並基於重構誤差判斷異常。"""
    model.eval()
    reconstruction_errors = []

    with torch.no_grad():
        for data in tqdm(data_loader, desc="Predicting"):
            data = data.to(device)
            output = model(data)
            # 計算均方誤差作為重構誤差
            error = torch.mean((output - data) ** 2, dim=1)
            reconstruction_errors.extend(error.cpu().numpy())

    # 異常判斷
    predictions = (np.array(reconstruction_errors) > threshold).astype(int)

    return predictions, np.array(reconstruction_errors)


if __name__ == "__main__":
    # 確保 log_data_generator.py 已經執行並生成了 log_file
    # 您可能需要先執行生成器腳本
    # import os
    # if not os.path.exists(LOG_FILE):
    #     print("Log file not found. Please run log_data_generator.py first.")
    #     exit()

    # --- 1. 資料加載與特徵工程 ---
    df = load_and_parse_logs(LOG_FILE)
    features_scaled, scaler, feature_names = feature_engineer(df)

    # --- 2. 準備訓練數據 ---
    # Autoencoder 僅使用 NORMAL 數據進行訓練
    normal_indices = df[df["is_anomaly"] == 0].index
    normal_features = features_scaled[normal_indices]

    # 劃分訓練集和驗證集 (為了設定合理的閾值)
    train_data, val_data = train_test_split(
        normal_features, test_size=0.2, random_state=42
    )

    train_dataset = LogDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    val_dataset = LogDataset(val_data)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    # --- 3. 模型訓練 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    INPUT_DIM = features_scaled.shape[1]
    print(f"Input dimension: {INPUT_DIM}")

    model = Autoencoder(INPUT_DIM, encoding_dim=16).to(device)
    train_model(model, train_loader, epochs=100, device=device)  # 訓練 50 個 Epoch

    # --- 4. 設定異常閾值 ---
    # 計算在正常驗證集上的重構誤差
    _, val_errors = predict_anomalies(model, val_loader, device, threshold=0)

    # 設定閾值: 通常取正常數據重構誤差的 Q95 或 Q99
    # 這裡我們取 95th percentile
    threshold = np.quantile(val_errors, 0.995)
    print(f"\n設定異常重構誤差閾值 (95th percentile of Normal Data): {threshold:.4f}")

    # --- 5. 異常檢測 ---
    full_dataset = LogDataset(features_scaled)
    full_loader = DataLoader(full_dataset, batch_size=512, shuffle=False)

    predictions, errors = predict_anomalies(model, full_loader, device, threshold)

    # --- 6. 輸出結果與評估 ---
    df["pred_anomaly"] = predictions

    # 找出被標記為異常的連線
    anomalies_detected = df[df["pred_anomaly"] == 1]

    print("\n[!!!] Autoencoder IDS 檢測結果 [!!!]")
    print(f"總行數: {len(df)}")
    print(f"檢測到的異常數: {len(anomalies_detected)}")

    # 輸出前 20 筆檢測到的異常，並比較實際標籤
    print("\n前 20 筆檢測到的異常 (Pred vs. Actual):")
    for idx, row in anomalies_detected.head(20).iterrows():
        print(
            f"Line {idx+1} | IP: {row['ip']} | Request: {row['request'][:50]}... | Error: {errors[idx]:.4f} | Actual Label: {row['label']}"
        )

    # 獲取真實標籤 (y_true) 和模型預測標籤 (y_pred)
    y_true = df["is_anomaly"].values
    y_pred = df["pred_anomaly"].values

    # 獲取模型輸出的重構誤差 (作為異常分數，越大越異常)
    anomaly_scores = errors

    print("\n" + "=" * 50)
    print("           [模型性能評估報告]")
    print("=" * 50)

    # 1. 計算基本分類指標 (基於設定的閾值)

    # Accuracy (準確度)
    acc = accuracy_score(y_true, y_pred)
    print(f"1. Accuracy (ACC): {acc:.4f} (整體準確率)")

    # Precision (精確度) - 預測為異常的中有多少是真的異常
    precision = precision_score(y_true, y_pred, zero_division=0)
    print(f"2. Precision: {precision:.4f} (誤報率的倒數)")

    # Recall (召回率/查全率) - 實際為異常的中有多少被抓到
    recall = recall_score(y_true, y_pred, zero_division=0)
    print(f"3. Recall: {recall:.4f} (漏報率的倒數)")

    # F1-Score - Precision 和 Recall 的調和平均數
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"4. F1-Score: {f1:.4f} (綜合指標)")

    print("-" * 50)

    # 2. 計算曲線下面積指標 (基於連續分數)

    # ROC AUC (Receiver Operating Characteristic - Area Under Curve)
    # 衡量模型區分正負類的能力 (與閾值無關)
    try:
        roc_auc = roc_auc_score(y_true, anomaly_scores)
        print(f"5. ROC-AUC: {roc_auc:.4f} (越高越好, 衡量 TPR vs FPR)")
    except ValueError:
        print("5. ROC-AUC: 無法計算 (數據中只有單一類別或異常數量過少)")

    # PR-AUC (Precision-Recall - Area Under Curve)
    # 專門用於不平衡數據集，衡量 Precision vs Recall
    precision_points, recall_points, _ = precision_recall_curve(y_true, anomaly_scores)
    pr_auc = auc(recall_points, precision_points)
    print(f"6. PR-AUC: {pr_auc:.4f} (越高越好, 專注於正類預測)")

    print("=" * 50)
