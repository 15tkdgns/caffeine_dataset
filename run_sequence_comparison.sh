#!/bin/bash
# 시퀀스 모델 비교 - 샘플 데이터 테스트

echo "======================================================================"
echo "시퀀스 모델 비교 (샘플 10% 데이터)"
echo "모델: FT-Transformer, TabNet, GRU+Attention, TCN"
echo "======================================================================"

cd /root/ibm_data2

# 1. 가상환경 생성
echo -e "\n[1/5] 가상환경 설정"
VENV_DIR="/root/ibm_data2/venv_sequence"

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv $VENV_DIR
fi

source $VENV_DIR/bin/activate

pip install --upgrade pip -q
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q
pip install pytorch-tabnet scikit-learn pandas numpy einops -q

# 2. 모델 학습 및 비교
echo -e "\n[2/5] 모델 학습 시작"

python3 << 'EOF'
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import time
import json
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("시퀀스 모델 비교 (샘플 10% 데이터)")
print("="*70)

# GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# 데이터 로드 및 샘플링
print("\n[데이터 로드]")
X_full = np.load('02_data/02_augmented/X_train_smote.npy')
y_full = np.load('02_data/02_augmented/y_train_smote.npy')
X_test_full = np.load('02_data/02_augmented/X_test.npy')
y_test_full = np.load('02_data/02_augmented/y_test.npy')

# 10% 샘플링
sample_idx = np.random.choice(len(X_full), size=int(len(X_full)*0.1), replace=False)
X_train = X_full[sample_idx]
y_train = y_full[sample_idx]

test_idx = np.random.choice(len(X_test_full), size=int(len(X_test_full)*0.1), replace=False)
X_test = X_test_full[test_idx]
y_test = y_test_full[test_idx]

print(f"학습: {len(X_train):,}, 테스트: {len(X_test):,}")

n_features = X_train.shape[1]
n_classes = len(np.unique(y_train))
print(f"피처: {n_features}, 클래스: {n_classes}")

results = {}

# ============================================================
# 1. TabNet
# ============================================================
print("\n" + "="*70)
print("[1/4] TabNet")
print("="*70)

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    
    start = time.time()
    tabnet = TabNetClassifier(
        n_d=32, n_a=32, n_steps=3,
        gamma=1.3, n_independent=2, n_shared=2,
        lambda_sparse=1e-3,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        scheduler_params={"step_size": 10, "gamma": 0.9},
        mask_type='entmax',
        device_name='cuda',
        verbose=0
    )
    
    tabnet.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        max_epochs=30,
        patience=10,
        batch_size=2048,
        virtual_batch_size=256,
        drop_last=False
    )
    train_time = time.time() - start
    
    y_pred = tabnet.predict(X_test)
    
    results['TabNet'] = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'macro_f1': float(f1_score(y_test, y_pred, average='macro')),
        'train_time': train_time
    }
    print(f"✅ Accuracy: {results['TabNet']['accuracy']:.4f}, Macro F1: {results['TabNet']['macro_f1']:.4f} ({train_time:.1f}초)")
except Exception as e:
    print(f"❌ TabNet 실패: {e}")

# GPU 메모리 정리
torch.cuda.empty_cache()

# ============================================================
# 2. GRU + Attention
# ============================================================
print("\n" + "="*70)
print("[2/4] GRU + Attention")
print("="*70)

class GRUAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_classes, n_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, n_layers, batch_first=True, dropout=0.2)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, n_classes)
        )
    
    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)  # (batch, 1, hidden)
        x, _ = self.gru(x)
        x, _ = self.attention(x, x, x)
        x = x.squeeze(1)
        return self.fc(x)

try:
    model = GRUAttention(n_features, 128, n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    X_tensor = torch.FloatTensor(X_train).to(device)
    y_tensor = torch.LongTensor(y_train).to(device)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=2048, shuffle=True)
    
    start = time.time()
    model.train()
    for epoch in range(20):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    train_time = time.time() - start
    
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        outputs = model(X_test_tensor)
        y_pred = outputs.argmax(dim=1).cpu().numpy()
    
    results['GRU+Attention'] = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'macro_f1': float(f1_score(y_test, y_pred, average='macro')),
        'train_time': train_time
    }
    print(f"✅ Accuracy: {results['GRU+Attention']['accuracy']:.4f}, Macro F1: {results['GRU+Attention']['macro_f1']:.4f} ({train_time:.1f}초)")
except Exception as e:
    print(f"❌ GRU+Attention 실패: {e}")

torch.cuda.empty_cache()

# ============================================================
# 3. TCN (Temporal Convolutional Network)
# ============================================================
print("\n" + "="*70)
print("[3/4] TCN (Temporal CNN)")
print("="*70)

class TCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_classes, kernel_size=3):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, n_classes)
        )
    
    def forward(self, x):
        x = self.embedding(x).unsqueeze(2)  # (batch, hidden, 1)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = x.squeeze(2)
        return self.fc(x)

try:
    model = TCN(n_features, 128, n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    X_tensor = torch.FloatTensor(X_train).to(device)
    y_tensor = torch.LongTensor(y_train).to(device)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=2048, shuffle=True)
    
    start = time.time()
    model.train()
    for epoch in range(20):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    train_time = time.time() - start
    
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        outputs = model(X_test_tensor)
        y_pred = outputs.argmax(dim=1).cpu().numpy()
    
    results['TCN'] = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'macro_f1': float(f1_score(y_test, y_pred, average='macro')),
        'train_time': train_time
    }
    print(f"✅ Accuracy: {results['TCN']['accuracy']:.4f}, Macro F1: {results['TCN']['macro_f1']:.4f} ({train_time:.1f}초)")
except Exception as e:
    print(f"❌ TCN 실패: {e}")

torch.cuda.empty_cache()

# ============================================================
# 4. Simple Transformer
# ============================================================
print("\n" + "="*70)
print("[4/4] Simple Transformer")
print("="*70)

class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_classes, n_heads=4, n_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim*4,
            dropout=0.2, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, n_classes)
        )
    
    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)  # (batch, 1, hidden)
        x = self.transformer(x)
        x = x.squeeze(1)
        return self.fc(x)

try:
    model = SimpleTransformer(n_features, 128, n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    X_tensor = torch.FloatTensor(X_train).to(device)
    y_tensor = torch.LongTensor(y_train).to(device)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=2048, shuffle=True)
    
    start = time.time()
    model.train()
    for epoch in range(20):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    train_time = time.time() - start
    
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        outputs = model(X_test_tensor)
        y_pred = outputs.argmax(dim=1).cpu().numpy()
    
    results['Transformer'] = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'macro_f1': float(f1_score(y_test, y_pred, average='macro')),
        'train_time': train_time
    }
    print(f"✅ Accuracy: {results['Transformer']['accuracy']:.4f}, Macro F1: {results['Transformer']['macro_f1']:.4f} ({train_time:.1f}초)")
except Exception as e:
    print(f"❌ Transformer 실패: {e}")

# 결과 저장
import os
os.makedirs('03_models/sequence_comparison', exist_ok=True)
with open('03_models/sequence_comparison/sample_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# 결과 출력
print("\n" + "="*70)
print("🏆 시퀀스 모델 비교 결과 (샘플 10%)")
print("="*70)

sorted_results = sorted(results.items(), key=lambda x: x[1]['macro_f1'], reverse=True)

print(f"\n{'순위':<4} {'모델':<20} {'Accuracy':>10} {'Macro F1':>10} {'시간':>8}")
print("-"*60)

for i, (name, metrics) in enumerate(sorted_results, 1):
    medal = "🥇" if i == 1 else ("🥈" if i == 2 else ("🥉" if i == 3 else "  "))
    print(f"{medal} {i}  {name:<20} {metrics['accuracy']:>10.4f} {metrics['macro_f1']:>10.4f} {metrics['train_time']:>7.1f}초")

print("\n" + "="*70)
print("완료!")
print("="*70)
EOF

deactivate

echo -e "\n결과: 03_models/sequence_comparison/sample_results.json"
