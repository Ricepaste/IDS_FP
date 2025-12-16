import pandas as pd
import numpy as np
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc
from tqdm import tqdm

# ==========================================
# 1. 設定與常數
# ==========================================
LOG_FILE = "apache_log_dataset.log"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定義規則特徵 (與生成器一致)
ATTACK_SIGNATURES = [
    r"(\s(or|union|select)\s.*=|'|--|/\*|information_schema)",  # SQL Injection
    r"(\.\./|\.\.\\|/etc/passwd|/windows/win\.ini)",            # Path Traversal
    r"(php://filter|base64-encode)",                            # LFI
]
MALICIOUS_AGENTS = [
    "sqlmap", "Nikto", "masscan", "Go-http-client", "Airlock"
]

# ==========================================
# 2. 工具函數
# ==========================================

def load_and_parse_logs(log_file):
    """讀取並解析日誌"""
    print("正在讀取並解析日誌...")
    data = []
    log_pattern = re.compile(
        r'^(?P<ip>\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}) - - \[(?P<time>.*?)\] '
        r'"(?P<request>.*?)" (?P<status>\d+) (?P<bytes>\d+) '
        r'"(?P<referer>.*?)" "(?P<user_agent>.*?)" \| (?P<label>.*)'
    )
    
    with open(log_file, 'r') as f:
        for line in f:
            match = log_pattern.match(line)
            if match:
                d = match.groupdict()
                parts = d['request'].split(' ')
                d['method'] = parts[0] if len(parts) > 0 else 'UNKNOWN'
                d['uri'] = parts[1] if len(parts) > 1 else 'UNKNOWN'
                data.append(d)
    
    df = pd.DataFrame(data)
    df['is_anomaly'] = df['label'].apply(lambda x: 0 if x == 'NORMAL' else 1)
    df['status'] = pd.to_numeric(df['status'])
    df['bytes'] = pd.to_numeric(df['bytes'])
    return df

def rule_based_check(row):
    """規則檢測器 (返回 1 為異常, 0 為正常)"""
    # 檢查 Request 內容
    for pattern in ATTACK_SIGNATURES:
        if re.search(pattern, row['request'], re.IGNORECASE):
            return 1
    # 檢查 User-Agent
    for agent in MALICIOUS_AGENTS:
        if agent.lower() in row['user_agent'].lower():
            return 1
    return 0

def process_features(df, top_ips=None, top_uas=None, train_columns=None):
    """
    特徵處理函數。
    如果是訓練階段：計算 top_ips, top_uas 並返回。
    如果是測試階段：使用傳入的 top_ips, top_uas, train_columns 確保一致性。
    """
    df_proc = df.copy()
    
    # 數值特徵
    df_proc['uri_len'] = df_proc['uri'].apply(lambda x: len(x.split('/')))
    
    # 類別特徵處理 (IP)
    if top_ips is None:
        top_ips = df_proc['ip'].value_counts().head(20).index.tolist()
    df_proc['ip_cat'] = df_proc['ip'].apply(lambda x: x if x in top_ips else 'OTHER_IP')
    
    # 類別特徵處理 (User-Agent - 簡化版)
    def categorize_ua(ua):
        if 'Mozilla' in ua and 'Chrome' in ua: return 'Chrome'
        if 'Mozilla' in ua and 'Safari' in ua: return 'Safari'
        if 'python-requests' in ua or 'curl' in ua: return 'Tool'
        if any(tool in ua for tool in ['sqlmap', 'Nikto', 'masscan']): return 'Malicious_Tool'
        return 'Other_UA'
    
    # 為了保持特徵一致性，我們不直接依賴數據中的 UA 分布，而是使用固定的類別邏輯
    # 但為了 One-Hot Encoding，我們仍需處理
    df_proc['ua_cat'] = df_proc['user_agent'].apply(categorize_ua)

    # One-Hot Encoding
    # 修正：先將 status 轉為字串，確保 get_dummies 會對其進行編碼
    df_proc['status'] = df_proc['status'].astype(str)
    
    # One-Hot Encoding
    df_encoded = pd.get_dummies(
        df_proc[['status', 'method', 'ip_cat', 'ua_cat']], 
        prefix=['status', 'method', 'ip', 'ua'],
        dtype=int  # 確保編碼後是 0/1 整數
    )
    
    # 合併數值特徵
    features = pd.concat([df_proc[['bytes', 'uri_len']], df_encoded], axis=1)
    
    # --- 關鍵：確保測試集擁有與訓練集完全相同的欄位 ---
    if train_columns is not None:
        # 補缺失的欄位 (填 0)
        for col in train_columns:
            if col not in features.columns:
                features[col] = 0
        # 移除多餘的欄位 (例如測試集中出現了訓練集沒見過的 IP)
        features = features[train_columns]
    else:
        train_columns = features.columns.tolist()
        
    return features, top_ips, train_columns

class LogDataset(Dataset):
    def __init__(self, dataframe):
        self.data = torch.tensor(dataframe.values, dtype=torch.float32)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# ==========================================
# 3. 主程式邏輯
# ==========================================

if __name__ == "__main__":
    # --- Step 1: 讀取並分割數據 (Train/Test Split) ---
    full_df = load_and_parse_logs(LOG_FILE)
    
    # 80% 訓練集, 20% 測試集
    # random_state 確保可重現性
    train_df, test_df = train_test_split(full_df, test_size=0.2, stratify=full_df['is_anomaly'])
    
    print(f"資料分割完成: 訓練集 {len(train_df)} 筆, 測試集 {len(test_df)} 筆")

    # --- Step 2: 訓練集特徵工程 (Fit) ---
    print("\n[Stage 1] 處理訓練集特徵...")
    X_train_raw, top_ips, train_cols = process_features(train_df)
    
    # 標準化 (Fit 只在訓練集上做!)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    
    # --- Step 3: 訓練集規則篩選 (準備 Autoencoder 數據) ---
    # Autoencoder 只能學習 "通過了規則檢查" 且 "確實是正常" 的數據
    
    # 1. 對訓練集應用規則
    train_rule_anomalies = train_df.apply(rule_based_check, axis=1)
    
    # 2. 篩選出純淨的正常數據 (Rule says Normal AND Label is Normal)
    # 我們不希望 AE 學習到被規則漏掉的異常，也不希望學習規則已經能抓到的異常
    ae_train_mask = (train_rule_anomalies == 0) & (train_df['is_anomaly'] == 0)
    
    X_ae_train = X_train_scaled[ae_train_mask]
    
    print(f"Autoencoder 訓練數據量 (Clean Normal Data): {len(X_ae_train)}")

    # --- Step 4: 訓練 Autoencoder ---
    print("\n[Stage 2] 訓練 Autoencoder...")
    input_dim = X_ae_train.shape[1]
    model = Autoencoder(input_dim).to(DEVICE)
    
    train_loader = DataLoader(LogDataset(pd.DataFrame(X_ae_train)), batch_size=64, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    model.train()
    EPOCHS = 20
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    print("Autoencoder 訓練完成。")

    # --- Step 5: 設定閾值 (使用訓練集的分佈) ---
    model.eval()
    with torch.no_grad():
        # 計算訓練數據的重構誤差
        train_tensor = torch.tensor(X_ae_train, dtype=torch.float32).to(DEVICE)
        reconstructions = model(train_tensor)
        train_loss = torch.mean((reconstructions - train_tensor) ** 2, dim=1).cpu().numpy()
    
    # 設定閾值為訓練集誤差的 99.9 分位數
    threshold = np.quantile(train_loss, 0.999)
    print(f"設定重構誤差閾值: {threshold:.6f}")

    # ==========================================
    # 下面開始針對 "測試集" 進行嚴格的評估
    # ==========================================
    print("\n" + "="*50)
    print("       [測試集 (Test Set) 評估階段]")
    print("="*50)

    # --- Step 6: 測試集特徵工程 (Transform Only) ---
    # 注意：使用訓練集定義的 top_ips 和 train_cols
    X_test_raw, _, _ = process_features(test_df, top_ips=top_ips, train_columns=train_cols)
    
    # 注意：使用訓練好的 scaler 進行 transform
    X_test_scaled = scaler.transform(X_test_raw)

    # --- Step 7: 混合模型推論 ---
    
    # 1. Rule-Based 預測
    y_pred_rule = test_df.apply(rule_based_check, axis=1).values
    
    # 2. Autoencoder 預測
    test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        test_recon = model(test_tensor)
        test_errors = torch.mean((test_recon - test_tensor) ** 2, dim=1).cpu().numpy()
    
    # AE 判定異常：誤差大於閾值 (threshold 在前面已計算)
    y_pred_ae = (test_errors > threshold).astype(int)
    
    # 3. Hybrid 預測 (聯集邏輯: 只要有一個說是異常，就是異常)
    y_pred_hybrid = y_pred_rule | y_pred_ae

    # 真實標籤
    y_true = test_df['is_anomaly'].values

    # --- Step 8: 建立比較報表 ---
    
    # 定義計算指標的函數
    def get_metrics(y_true, y_pred):
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1-Score": f1_score(y_true, y_pred, zero_division=0)
        }

    # 計算三種模型的指標
    metrics_rule = get_metrics(y_true, y_pred_rule)
    metrics_ae = get_metrics(y_true, y_pred_ae)
    metrics_hybrid = get_metrics(y_true, y_pred_hybrid)

    # 轉成 DataFrame 方便顯示
    results_df = pd.DataFrame([metrics_rule, metrics_ae, metrics_hybrid], 
                              index=["Rule-Based", "Autoencoder", "Hybrid IDS"])
    
    print("\n" + "="*60)
    print("             [三種模型性能評估比較表]")
    print("="*60)
    print(results_df.round(4))
    print("="*60)

    # --- Step 9: 視覺化繪圖 (更直觀的分析) ---
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    # 設定繪圖風格
    sns.set_style("whitegrid")
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] # 讓中文正常顯示 (Windows)
    plt.rcParams['axes.unicode_minus'] = False

    fig = plt.figure(figsize=(18, 10))

    # 1. 繪製指標長條圖 (Grouped Bar Chart)
    ax1 = plt.subplot(2, 1, 1)
    results_df.plot(kind='bar', ax=ax1, rot=0)
    ax1.set_title("各模型性能指標比較", fontsize=16)
    ax1.set_ylim(0, 1.1)
    ax1.legend(loc='lower right')
    
    # 在柱狀圖上標示數值
    for container in ax1.containers:
        ax1.bar_label(container, fmt='%.4f')

    # 2. 繪製混淆矩陣 (Confusion Matrices)
    # 定義 helper function
    def plot_cm(y_true, y_pred, ax, title):
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                    xticklabels=['正常', '異常'], yticklabels=['正常', '異常'])
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('預測標籤')
        ax.set_ylabel('真實標籤')

    ax2 = plt.subplot(2, 3, 4)
    plot_cm(y_true, y_pred_rule, ax2, "Rule-Based (規則)")

    ax3 = plt.subplot(2, 3, 5)
    plot_cm(y_true, y_pred_ae, ax3, "Autoencoder (AI)")

    ax4 = plt.subplot(2, 3, 6)
    plot_cm(y_true, y_pred_hybrid, ax4, "Hybrid (混合)")

    plt.tight_layout()
    plt.show()
    
    print("\n[分析提示]")
    print("1. Rule-Based: 通常 Precision 極高，但 Recall 較低 (無法抓到未知攻擊)。")
    print("2. Autoencoder: Recall 通常較高 (能抓異常行為)，但可能有較多誤報 (FP)。")
    print("3. Hybrid: 目標是結合兩者優勢，提升 Recall 的同時維持可接受的 Precision。")