"""
시퀀스 데이터 생성 + 시퀀스 모델 학습
사용자별 최근 N건 거래 이력 → 다음 카테고리 예측
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import time
import json
import os
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("시퀀스 데이터 생성 + 시퀀스 모델 학습")
print("="*70)

# GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ============================================================
# 1. 원본 데이터에서 시퀀스 생성
# ============================================================
print("\n[1/5] 원본 데이터 로드")

# 원본 데이터 로드 (시간 순서 유지)
raw_file = '02_data/00_raw/credit_card_transactions-ibm_v2.csv'
df_raw = pd.read_csv(raw_file)
print(f"  원본 데이터: {len(df_raw):,}건")

# 날짜 정보 생성
df_raw['Date'] = pd.to_datetime(
    df_raw['Year'].astype(str) + '-' + 
    df_raw['Month'].astype(str).str.zfill(2) + '-' + 
    df_raw['Day'].astype(str).str.zfill(2) + ' ' + 
    df_raw['Time']
)

# MCC → 카테고리 매핑
mcc_to_category = {
    range(5411, 5500): '식료품', range(5811, 5900): '외식',
    range(5200, 5300): '쇼핑', range(5300, 5400): '쇼핑', range(5600, 5700): '쇼핑',
    range(5500, 5600): '주유', range(4000, 4100): '교통', range(4100, 4200): '교통',
    range(4800, 4900): '생활', range(6000, 6100): '생활'
}

def get_category(mcc):
    for mcc_range, cat in mcc_to_category.items():
        if mcc in mcc_range:
            return cat
    return None

df_raw['Category'] = df_raw['MCC'].apply(get_category)
df_raw = df_raw[df_raw['Category'].notna()].copy()
print(f"  카테고리 매핑 후: {len(df_raw):,}건")

# 카테고리 인코딩
cat_to_idx = {cat: i for i, cat in enumerate(['식료품', '외식', '쇼핑', '주유', '교통', '생활'])}
df_raw['Category_idx'] = df_raw['Category'].map(cat_to_idx)

# 금액 정규화
df_raw['Amount_clean'] = df_raw['Amount'].replace(r'[\$,]', '', regex=True).astype(float)
amount_mean = df_raw['Amount_clean'].mean()
amount_std = df_raw['Amount_clean'].std()
df_raw['Amount_scaled'] = (df_raw['Amount_clean'] - amount_mean) / amount_std

# 시간 피처
df_raw['Hour'] = pd.to_datetime(df_raw['Time']).dt.hour
df_raw['DayOfWeek'] = df_raw['Date'].dt.dayofweek
df_raw['Hour_scaled'] = df_raw['Hour'] / 23.0
df_raw['DayOfWeek_scaled'] = df_raw['DayOfWeek'] / 6.0

# 사용자별 시간순 정렬
df_raw = df_raw.sort_values(['User', 'Date']).reset_index(drop=True)

# ============================================================
# 2. 시퀀스 데이터 생성
# ============================================================
print("\n[2/5] 시퀀스 데이터 생성")

SEQ_LENGTH = 10  # 최근 10건 거래로 다음 예측
FEATURES = ['Amount_scaled', 'Hour_scaled', 'DayOfWeek_scaled', 'Category_idx']

sequences = []
targets = []

# 사용자별 시퀀스 생성
users = df_raw['User'].unique()
print(f"  총 사용자: {len(users)}명")

for user in users:
    user_data = df_raw[df_raw['User'] == user][FEATURES].values
    
    # 시퀀스 생성 (슬라이딩 윈도우)
    for i in range(len(user_data) - SEQ_LENGTH):
        seq = user_data[i:i+SEQ_LENGTH]
        target = int(user_data[i+SEQ_LENGTH, 3])  # Category_idx
        sequences.append(seq)
        targets.append(target)

X = np.array(sequences, dtype=np.float32)
y = np.array(targets, dtype=np.int64)

print(f"  시퀀스 생성 완료")
print(f"  X shape: {X.shape} (샘플, 시퀀스길이, 피처)")
print(f"  y shape: {y.shape}")

# 샘플링 (10%로 빠른 테스트)
sample_idx = np.random.choice(len(X), size=min(500000, len(X)), replace=False)
X_sampled = X[sample_idx]
y_sampled = y[sample_idx]

print(f"  샘플링 후: {len(X_sampled):,}건")

# Train/Test 분할
X_train, X_test, y_train, y_test = train_test_split(
    X_sampled, y_sampled, test_size=0.2, random_state=42, stratify=y_sampled
)

print(f"  학습: {len(X_train):,}, 테스트: {len(X_test):,}")

# ============================================================
# 3. 시퀀스 모델 정의
# ============================================================
print("\n[3/5] 모델 정의")

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_classes, n_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, 
                           batch_first=True, dropout=0.3, bidirectional=True)
        self.attention = nn.MultiheadAttention(hidden_dim*2, num_heads=4, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, n_classes)
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        out = attn_out[:, -1, :]  # 마지막 시퀀스
        return self.fc(out)


class GRUClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_classes, n_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, 
                         batch_first=True, dropout=0.3, bidirectional=True)
        self.attention = nn.MultiheadAttention(hidden_dim*2, num_heads=4, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, n_classes)
        )
    
    def forward(self, x):
        gru_out, _ = self.gru(x)
        attn_out, _ = self.attention(gru_out, gru_out, gru_out)
        out = attn_out[:, -1, :]
        return self.fc(out)


class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                             padding=(kernel_size-1)*dilation, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        out = self.conv(x)
        out = out[:, :, :x.size(2)]  # 패딩 제거
        out = self.bn(out)
        out = self.relu(out)
        return self.dropout(out)


class TCNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_classes, n_layers=3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.tcn_blocks = nn.ModuleList([
            TCNBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=2**i)
            for i in range(n_layers)
        ])
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, n_classes)
        )
    
    def forward(self, x):
        x = self.input_proj(x)  # (batch, seq, hidden)
        x = x.transpose(1, 2)   # (batch, hidden, seq)
        for block in self.tcn_blocks:
            x = block(x)
        x = x[:, :, -1]         # 마지막 시퀀스
        return self.fc(x)

# ============================================================
# 4. 모델 학습
# ============================================================
print("\n[4/5] 모델 학습")

n_features = X_train.shape[2]
n_classes = len(np.unique(y_train))

results = {}

def train_model(model, name, epochs=30):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    
    X_tensor = torch.FloatTensor(X_train).to(device)
    y_tensor = torch.LongTensor(y_train).to(device)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=1024, shuffle=True)
    
    start = time.time()
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step(total_loss)
        
        if (epoch+1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")
    
    train_time = time.time() - start
    
    # 평가
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        outputs = model(X_test_tensor)
        y_pred = outputs.argmax(dim=1).cpu().numpy()
    
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    
    return {
        'accuracy': float(acc),
        'macro_f1': float(macro_f1),
        'train_time': train_time
    }

# 모델 학습
models = {
    'LSTM+Attention': LSTMClassifier(n_features, 128, n_classes).to(device),
    'GRU+Attention': GRUClassifier(n_features, 128, n_classes).to(device),
    'TCN': TCNClassifier(n_features, 128, n_classes).to(device)
}

for name, model in models.items():
    print(f"\n{'='*70}")
    print(f"[{name}]")
    print("="*70)
    
    try:
        result = train_model(model, name)
        results[name] = result
        print(f"✅ Accuracy: {result['accuracy']:.4f}, Macro F1: {result['macro_f1']:.4f} ({result['train_time']:.1f}초)")
    except Exception as e:
        print(f"❌ 실패: {e}")
    
    torch.cuda.empty_cache()

# ============================================================
# 5. 결과
# ============================================================
print("\n" + "="*70)
print("🏆 시퀀스 모델 비교 결과 (실제 시퀀스 데이터)")
print("="*70)

# 결과 저장
os.makedirs('03_models/sequence_comparison', exist_ok=True)
with open('03_models/sequence_comparison/sequence_results.json', 'w') as f:
    json.dump(results, f, indent=2)

sorted_results = sorted(results.items(), key=lambda x: x[1]['macro_f1'], reverse=True)

print(f"\n{'순위':<4} {'모델':<20} {'Accuracy':>10} {'Macro F1':>10} {'시간':>8}")
print("-"*60)

for i, (name, metrics) in enumerate(sorted_results, 1):
    medal = "🥇" if i == 1 else ("🥈" if i == 2 else "🥉")
    print(f"{medal} {i}  {name:<20} {metrics['accuracy']:>10.4f} {metrics['macro_f1']:>10.4f} {metrics['train_time']:>7.1f}초")

print("\n" + "="*70)
print("완료!")
print("="*70)
