import random
from datetime import datetime, timedelta
from faker import Faker
from tqdm import tqdm  # 用於顯示進度條

# 初始化 Faker
fake = Faker()

# 惡意 User-Agents 列表
MALICIOUS_AGENTS = [
    "sqlmap/1.5.13#stable",
    "Nikto/2.1.6",
    "masscan/1.0",
    "Go-http-client/1.1",
    "Airlock/2.0",
]

# 惡意請求模式
ATTACK_PATTERNS = {
    "SQL_INJECTION": [
        "GET /products.php?id=1%20OR%201=1-- HTTP/1.1",
        "GET /login.php?user=' UNION SELECT password FROM users -- HTTP/1.1",
        "POST /search.asp?q=test%27%20OR%20%271%27=%271 HTTP/1.1",
    ],
    "PATH_TRAVERSAL": [
        "GET /show.php?page=../etc/passwd HTTP/1.1",
        "GET /config.php?file=../../../../windows/win.ini HTTP/1.1",
        "GET /image.jpg?file=../../../boot.ini HTTP/1.1",
    ],
    "LFI_ATTEMPT": [
        "GET /index.php?page=file:///etc/passwd HTTP/1.1",
        "GET /?file=php://filter/read=convert.base64-encode/resource=/etc/hosts HTTP/1.1",
    ],
}


def generate_log_line(start_time: datetime, anomaly_type: str = "NORMAL") -> str:
    """生成單一 Apache Combined Log Format 行，並標記異常類型。"""

    # 共同欄位
    ip = fake.ipv4()
    log_time = start_time.strftime("%d/%b/%Y:%H:%M:%S +0800")

    # 正常連線設定
    if anomaly_type == "NORMAL":
        method = random.choice(["GET", "POST", "HEAD"])
        uri = fake.uri_path()
        request = f'"{method} {uri} HTTP/1.1"'
        status = random.choice(
            [200, 200, 200, 200, 201, 302, 404]
        )  # 正常情況下 200 最多
        bytes_sent = random.randint(100, 20000)
        referer = random.choice(
            [fake.url(), "-", "https://google.com", "https://app.com"]
        )
        user_agent = fake.user_agent()

    # 惡意掃描工具連線設定
    elif anomaly_type == "SCANNER_UA":
        request = f'"GET /{fake.word()}.php HTTP/1.1"'
        status = random.choice([403, 404])
        bytes_sent = random.randint(100, 500)
        referer = "-"
        user_agent = random.choice(MALICIOUS_AGENTS)

    # 內容型攻擊連線設定
    elif anomaly_type in ATTACK_PATTERNS:
        request = f'"{random.choice(ATTACK_PATTERNS[anomaly_type])}"'
        status = random.choice([400, 403, 500])  # 攻擊通常會產生錯誤或被 WAF 攔截
        bytes_sent = random.randint(100, 500)
        referer = "-"
        user_agent = random.choice(
            [fake.user_agent(), "python-requests/2.25.1", "curl/7.64.1"]
        )

    # 組合日誌行
    log_line = (
        f"{ip} - - [{log_time}] {request} {status} {bytes_sent} "
        f'"{referer}" "{user_agent}" | {anomaly_type}\n'
    )
    return log_line


def generate_dataset(filename: str, num_lines: int):
    """生成指定行數的日誌資料集，並寫入檔案。"""

    start_time = datetime(2025, 12, 15, 10, 0, 0)

    # 設定異常機率
    # 90% 正常, 10% 異常
    NORMAL_PROB = 0.90
    ATTACK_TYPES = list(ATTACK_PATTERNS.keys()) + ["SCANNER_UA"]

    with open(filename, "w") as f:
        # 使用 tqdm 顯示進度
        for i in tqdm(range(num_lines), desc="Generating Logs"):
            current_time = start_time + timedelta(seconds=i)

            # 隨機決定是否為異常
            if random.random() < NORMAL_PROB:
                log_line = generate_log_line(current_time, "NORMAL")
            else:
                # 隨機選擇一種異常類型
                anomaly_type = random.choice(ATTACK_TYPES)
                log_line = generate_log_line(current_time, anomaly_type)

            f.write(log_line)


# --- 執行生成 ---
if __name__ == "__main__":
    DATASET_SIZE = 100000  # 您可以改成更大的數字，例如 100000 或 1000000
    OUTPUT_FILE = "apache_log_dataset.log"
    print(f"開始生成 {DATASET_SIZE} 行日誌數據到 {OUTPUT_FILE}...")
    generate_dataset(OUTPUT_FILE, DATASET_SIZE)
    print("日誌數據生成完成！")
