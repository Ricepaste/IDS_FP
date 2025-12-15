import re
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

# 1. 定義惡意簽名
ATTACK_SIGNATURES = [
    r"(\s(or|union|select)\s.*=|'|--|/\*|information_schema)",  # SQL Injection 關鍵字
    r"(\.\./|\.\.\\|/etc/passwd|/windows/win\.ini)",  # Path Traversal 關鍵字
    r"(php://filter|base64-encode)",  # Local File Inclusion (LFI)
    r"(\bcmd\.exe\b|\bsh\b|\bcat\b|\bperl\b)",  # OS Command Injection 關鍵字
]

# 2. 定義惡意 User-Agents
MALICIOUS_AGENTS = ["sqlmap", "Nikto", "masscan", "Go-http-client", "Airlock"]

# 3. Apache Combined Log 正則表達式
# 捕捉 IP, 時間, 請求, 狀態碼, User-Agent
LOG_PATTERN = re.compile(
    r"^(?P<ip>\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}) - - \[(?P<time>.*?)\] "
    r'"(?P<request>.*?)" (?P<status>\d+) (?P<bytes>\d+) '
    r'"(?P<referer>.*?)" "(?P<user_agent>.*?)"'
)

# 頻率異常檢測的參數
RATE_LIMIT_THRESHOLD = 50  # 1 分鐘內超過 50 個錯誤請求
TIME_WINDOW_SECONDS = 60  # 1 分鐘的時間窗口


def parse_log_line(line: str) -> Dict:
    """解析單行日誌，返回字典格式的資料。"""
    match = LOG_PATTERN.match(line)
    if match:
        data = match.groupdict()
        # 轉換時間格式
        try:
            data["datetime"] = datetime.strptime(
                data["time"].split(" ")[0], "%d/%b/%Y:%H:%M:%S"
            )
        except ValueError:
            data["datetime"] = datetime.min  # 無效時間設為最小
        data["status"] = int(data["status"])
        return data
    return {}


def detect_anomalies(log_file: str):
    """讀取日誌檔並執行異常檢測。"""

    anomalies: List[Tuple[str, int, str]] = []  # 儲存 (IP, 行號, 異常類型)

    # 紀錄每個 IP 在特定時間窗口內的錯誤請求數量
    ip_error_count: Dict[str, List[Tuple[datetime, int]]] = defaultdict(list)

    with open(log_file, "r") as f:
        for line_num, line in enumerate(f, 1):
            log_data = parse_log_line(line)
            if not log_data:
                continue

            ip = log_data["ip"]
            request = log_data["request"]
            user_agent = log_data["user_agent"]
            status = log_data["status"]
            current_time = log_data["datetime"]

            # --- 規則 1: 惡意內容檢測 (Signature-Based) ---
            for pattern in ATTACK_SIGNATURES:
                if re.search(pattern, request, re.IGNORECASE):
                    anomalies.append((ip, line_num, "CONTENT_SIGNATURE_MATCH"))
                    break

            # --- 規則 2: 惡意代理檢測 (UA-Based) ---
            for agent in MALICIOUS_AGENTS:
                if agent.lower() in user_agent.lower():
                    anomalies.append((ip, line_num, "MALICIOUS_USER_AGENT"))
                    break

            # --- 規則 3: 頻率異常檢測 (Rate-Based) ---
            if 400 <= status <= 599:
                # 紀錄錯誤請求的時間戳
                ip_error_count[ip].append((current_time, status))

                # 過濾掉時間窗口外的舊紀錄
                valid_requests = [
                    (t, s)
                    for t, s in ip_error_count[ip]
                    if current_time - t <= timedelta(seconds=TIME_WINDOW_SECONDS)
                ]
                ip_error_count[ip] = valid_requests

                # 檢查是否超過閾值
                if len(valid_requests) >= RATE_LIMIT_THRESHOLD:
                    anomalies.append(
                        (
                            ip,
                            line_num,
                            f"HIGH_ERROR_RATE ({len(valid_requests)} in {TIME_WINDOW_SECONDS}s)",
                        )
                    )

    # --- 輸出結果 ---
    if anomalies:
        print("\n[!!!] 檢測到異常連線 [!!!]\n")
        print(f"{'IP':<15} | {'Line':<6} | {'Anomaly Type'}")
        print("-" * 40)
        # 為了避免重複檢測，對異常類型進行去重
        unique_anomalies = sorted(list(set(anomalies)), key=lambda x: x[1])
        for ip, line_num, anomaly_type in unique_anomalies:
            print(f"{ip:<15} | {line_num:<6} | {anomaly_type}")
    else:
        print("\n[✓] 未檢測到重大異常。")


# --- 執行 IDS ---
if __name__ == "__main__":
    LOG_FILE = "apache_log_dataset.log"

    # 執行 IDS
    print(f"開始執行簡易 IDS 檢測 {LOG_FILE}...")
    detect_anomalies(LOG_FILE)
