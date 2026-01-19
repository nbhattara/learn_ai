
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import warnings
import time
import random

warnings.filterwarnings("ignore", category=UserWarning)

# ───────────────── CONFIG ─────────────────
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

TEST_SIZE = 0.20
VALID_SIZE = 0.15
BATCH_SIZE = 64
EPOCHS = 300
BASE_LR = 1e-3
MIN_LR = 1e-6
PATIENCE = 40

DISEASE_MAP = {
    0: "Healthy",
    1: "Diabetes",
    2: "Coronary Artery Disease",
    3: "Hypertension"
}

# ─────────────── LOAD REAL DATA ───────────────
df = pd.read_csv("real_health_data.csv")

features = [
    'age','fasting_glucose','systolic_bp','diastolic_bp',
    'total_cholesterol','hdl_cholesterol','triglycerides','bmi'
]

X = df[features].values.astype(np.float32)
y = df['disease'].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval,
    test_size=VALID_SIZE/(1-TEST_SIZE),
    stratify=y_trainval,
    random_state=RANDOM_STATE
)

# ─────────────── DATASET ───────────────
class HealthDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).long()
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

train_loader = DataLoader(HealthDataset(X_train,y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(HealthDataset(X_val,y_val), batch_size=BATCH_SIZE*2)
test_loader  = DataLoader(HealthDataset(X_test,y_test), batch_size=BATCH_SIZE*2)

# ─────────────── MODEL ───────────────
class HealthNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8,256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.35),
            nn.Linear(256,128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.30),
            nn.Linear(128,64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.20),
            nn.Linear(64,32), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(32,4)
        )
    def forward(self,x): return self.net(x)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = HealthNet().to(DEVICE)

criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=2e-5)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=MIN_LR)

# ─────────────── TRAINING ───────────────
best_acc, patience = 0, 0
best_state = None

for epoch in range(EPOCHS):
    model.train()
    for Xb,yb in train_loader:
        Xb,yb = Xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(Xb), yb)
        loss.backward()
        optimizer.step()

    model.eval()
    preds,trues = [],[]
    with torch.no_grad():
        for Xb,yb in val_loader:
            out = model(Xb.to(DEVICE))
            preds += torch.argmax(out,1).cpu().tolist()
            trues += yb.tolist()

    acc = accuracy_score(trues,preds)
    scheduler.step()

    if acc > best_acc:
        best_acc = acc
        best_state = model.state_dict()
        patience = 0
    else:
        patience += 1

    if (epoch+1)%20==0:
        print(f"Epoch {epoch+1:3d} | Val Accuracy: {acc:.2%}")

    if patience >= PATIENCE:
        print("Early stopping")
        break

model.load_state_dict(best_state)

# ─────────────── EVALUATION ───────────────
model.eval()
preds,trues = [],[]
with torch.no_grad():
    for Xb,yb in test_loader:
        out = model(Xb.to(DEVICE))
        preds += torch.argmax(out,1).cpu().tolist()
        trues += yb.tolist()

print("\nTest Accuracy:", accuracy_score(trues,preds))
print(classification_report(trues,preds,target_names=DISEASE_MAP.values()))

# ─────────────── SAFE PREDICTION ───────────────
def predict_risk(values):
    X = scaler.transform([values])
    with torch.no_grad():
        probs = torch.softmax(model(torch.tensor(X,dtype=torch.float32).to(DEVICE)),1)[0].cpu().numpy()
    idx = int(np.argmax(probs))
    conf = probs[idx]*100
    risk = "Low" if conf<60 else "Moderate" if conf<80 else "High"
    return DISEASE_MAP[idx], risk, conf, probs

# ─────────────── CHATBOT ───────────────
print("\n⚠️ DISCLAIMER: This system estimates health risk patterns. It does NOT diagnose disease.\n")

while True:
    try:
        age = input("Age (or exit): ")
        if age.lower()=='exit': break
        vals = [float(age)] + [float(input(f)) for f in [
            "Fasting Glucose: ","Systolic BP: ","Diastolic BP: ","Total Cholesterol: ","HDL: ","Triglycerides: ","BMI: "]]

        disease,risk,conf,probs = predict_risk(vals)
        print("\nPrimary risk pattern:", disease)
        print("Overall risk level :", risk)
        print(f"Model confidence   : {conf:.1f}%")
        print("(Confidence reflects model certainty, not medical certainty)\n")

    except Exception as e:
        print("Invalid input",e)
