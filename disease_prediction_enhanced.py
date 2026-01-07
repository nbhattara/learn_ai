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
from datetime import datetime

warnings.filterwarnings("ignore", category=UserWarning)

# ────────────────────────────────────────────────────────────────
#                     CONFIGURATION
# ────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
torch.cuda.manual_seed_all(RANDOM_STATE)

TEST_SIZE = 0.20
VALID_SIZE = 0.15
BATCH_SIZE = 64
EPOCHS = 300
BASE_LR = 1e-3
MIN_LR = 1e-6
PATIENCE_EARLY_STOP = 40

DISEASE_MAP = {
    0: "Healthy",
    1: "Diabetes",
    2: "Coronary Artery Disease",
    3: "Hypertension"
}

RECOMMENDATIONS = {
    "Healthy": [
        "Maintain balanced nutrition rich in vegetables, fruits, whole grains and lean proteins",
        "Engage in at least 150–300 minutes of moderate aerobic activity per week",
        "Keep 7–9 hours of quality sleep nightly",
        "Perform annual preventive health check-ups including lipid profile and glucose",
        "Avoid all forms of tobacco and limit alcohol to recommended guidelines"
    ],
    "Diabetes": [
        "Strongly limit refined sugars, sweets and processed carbohydrates",
        "Monitor fasting, postprandial and HbA1c regularly",
        "Follow personalized low-glycemic or Mediterranean-style meal plan",
        "Aim for 150+ min/week moderate exercise + resistance training",
        "Maintain regular contact with diabetologist/endocrinologist",
        "Target healthy weight (BMI 18.5–22.9 for most Asian populations)"
    ],
    "Coronary Artery Disease": [
        "Follow strict heart-healthy diet (low saturated fat, high fiber, plant sterols)",
        "Achieve and maintain LDL <70 mg/dL (very high risk) or <100 mg/dL",
        "Complete smoking cessation – most important single modifiable factor",
        "Participate in supervised/structured cardiac rehabilitation program",
        "Strict control of BP (<130/80), glucose and lipids",
        "Consider low-dose aspirin/statins/ACEi/ARB per cardiologist"
    ],
    "Hypertension": [
        "Restrict dietary sodium to 1500–2000 mg/day (DASH / low-salt diet)",
        "Increase potassium-rich foods (banana, spinach, sweet potato, beans)",
        "Regular aerobic exercise 5–7 days/week (brisk walking, cycling, swimming)",
        "Home BP monitoring with validated upper-arm device (morning & evening)",
        "Limit alcohol ≤14 units/week men, ≤7 women",
        "Practice structured stress management (mindfulness, yoga, deep breathing)"
    ]
}

# ────────────────────────────────────────────────────────────────
#        IMPROVED & SAFE SYNTHETIC DATA GENERATION
# ────────────────────────────────────────────────────────────────
def generate_synthetic_health_data(n_samples=1200):
    np.random.seed(RANDOM_STATE)
    
    age = np.random.normal(52, 14, n_samples).clip(25, 82).astype(int)
    
    glucose = np.random.normal(np.where(age < 45, 92, 105), 18, n_samples)
    sbp = np.random.normal(118 + (age-30)*0.55, 16, n_samples)
    dbp = np.random.normal(76 + (age-30)*0.25, 10, n_samples)
    total_chol = np.random.normal(185 + (age-30)*0.9, 38, n_samples)
    
    hdl_base = np.random.choice([48,52,55,58,62], n_samples, p=[0.25,0.3,0.25,0.15,0.05])
    hdl = np.random.normal(hdl_base, 12, n_samples).clip(28, 98)
    
    # Safe triglycerides generation - using shifted lognormal
    tg_mean_log = np.log(130) + 0.08 * (age - 50) / 10
    triglycerides = np.random.lognormal(tg_mean_log, 0.58, n_samples)
    triglycerides = np.clip(triglycerides, 40, 1200)  # realistic range
    
    bmi = np.random.normal(26.5 + (age-35)*0.08, 5.2, n_samples).clip(17, 48)
    
    # Disease assignment (simplified hierarchical logic)
    disease = np.zeros(n_samples, dtype=int)
    
    # Diabetes
    diabetes_mask = (glucose >= 126) | ((glucose >= 110) & (np.random.random(n_samples) < 0.18))
    disease[diabetes_mask] = 1
    
    # Hypertension
    htn_mask = ((sbp >= 140) | (dbp >= 90)) & ~diabetes_mask
    htn_mask |= ((sbp >= 132) & (np.random.random(n_samples) < 0.22))
    disease[htn_mask] = 3
    
    # CAD
    cad_risk = (
        (age > 55) * 0.38 +
        (total_chol > 240) * 0.28 +
        (hdl < 40) * 0.28 +
        (bmi > 30) * 0.18 +
        (sbp > 145) * 0.22 +
        (disease == 1) * 0.32
    )
    cad_mask = (cad_risk > 0.72) | (np.random.random(n_samples) < cad_risk * 0.45)
    cad_mask &= ~diabetes_mask
    disease[cad_mask] = 2
    
    data = {
        'age': age,
        'fasting_glucose': np.round(glucose).astype(int),
        'systolic_bp': np.round(sbp).astype(int),
        'diastolic_bp': np.round(dbp).astype(int),
        'total_cholesterol': np.round(total_chol).astype(int),
        'hdl_cholesterol': np.round(hdl).astype(int),
        'triglycerides': np.round(triglycerides).astype(int),
        'bmi': np.round(bmi, 1),
        'disease': disease
    }
    
    return pd.DataFrame(data)

print("Generating realistic synthetic health dataset...")
df = generate_synthetic_health_data(1200)
print("\nClass distribution:")
print(df['disease'].value_counts().to_frame().rename(columns={'disease':'count'}))

# ────────────────────────────────────────────────────────────────
#                     DATA PREPARATION
# ────────────────────────────────────────────────────────────────
features = [
    'age', 'fasting_glucose', 'systolic_bp', 'diastolic_bp',
    'total_cholesterol', 'hdl_cholesterol', 'triglycerides', 'bmi'
]

X = df[features].values.astype(np.float32)
y = df['disease'].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=VALID_SIZE/(1-TEST_SIZE),
    stratify=y_trainval, random_state=RANDOM_STATE
)

# ────────────────────────────────────────────────────────────────
#                        PYTORCH DATASET
# ────────────────────────────────────────────────────────────────
class HealthDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).long()
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = HealthDataset(X_train, y_train)
val_ds   = HealthDataset(X_val, y_val)
test_ds  = HealthDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE*2, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE*2, shuffle=False)

# ────────────────────────────────────────────────────────────────
#                     IMPROVED MODEL
# ────────────────────────────────────────────────────────────────
class AdvancedHealthClassifier(nn.Module):
    def __init__(self, input_size=8, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.35),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.30),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.20),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.15),
            
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)

# ────────────────────────────────────────────────────────────────
#                    DEVICE & MODEL SETUP
# ────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

model = AdvancedHealthClassifier(input_size=len(features)).to(DEVICE)

criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=2e-5)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=MIN_LR)

# ────────────────────────────────────────────────────────────────
#                        TRAINING
# ────────────────────────────────────────────────────────────────
print(f"\nTraining advanced model on {DEVICE} with {len(df):,} samples...\n")

best_val_acc = 0.0
best_epoch = 0
patience_counter = 0
best_state_dict = None

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model(Xb)
        loss = criterion(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_loss += loss.item()
        _, pred = torch.max(logits, 1)
        train_total += yb.size(0)
        train_correct += (pred == yb).sum().item()
    
    train_loss /= len(train_loader)
    train_acc = train_correct / train_total
    
    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            logits = model(Xb)
            loss = criterion(logits, yb)
            val_loss += loss.item()
            _, pred = torch.max(logits, 1)
            val_total += yb.size(0)
            val_correct += (pred == yb).sum().item()
    
    val_loss /= len(val_loader)
    val_acc = val_correct / val_total
    
    scheduler.step()
    
    if (epoch + 1) % 20 == 0 or epoch == EPOCHS-1:
        print(f"[{epoch+1:3d}/{EPOCHS}]  Train: {train_loss:.4f} ({train_acc:.2%})  "
              f"Val: {val_loss:.4f} ({val_acc:.2%})  LR: {optimizer.param_groups[0]['lr']:.2e}")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch + 1
        best_state_dict = model.state_dict()
        patience_counter = 0
    else:
        patience_counter += 1
    
    if patience_counter >= PATIENCE_EARLY_STOP:
        print(f"\nEarly stopping after {epoch+1} epochs.")
        break

if best_state_dict is not None:
    model.load_state_dict(best_state_dict)
print(f"\nBest validation accuracy: {best_val_acc:.2%} @ epoch {best_epoch}\n")

# ────────────────────────────────────────────────────────────────
#                      FINAL EVALUATION
# ────────────────────────────────────────────────────────────────
def evaluate_model(loader, name="Set"):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(DEVICE)
            logits = model(Xb)
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            preds.extend(pred)
            trues.extend(yb.numpy())
    
    acc = accuracy_score(trues, preds)
    print(f"\n{name} Performance:")
    print(f"Accuracy: {acc:.3%}")
    print("\nClassification Report:")
    print(classification_report(trues, preds, target_names=list(DISEASE_MAP.values()), digits=3))

evaluate_model(test_loader, "Test")

# ────────────────────────────────────────────────────────────────
#                      PREDICTION FUNCTION
# ────────────────────────────────────────────────────────────────
def predict_patient(**kwargs):
    required = ['age', 'fasting_glucose', 'systolic_bp', 'diastolic_bp',
                'total_cholesterol', 'hdl_cholesterol', 'triglycerides', 'bmi']
    
    values = [kwargs[k] for k in required]
    arr = np.array([values], dtype=np.float32)
    scaled = scaler.transform(arr)
    tensor = torch.from_numpy(scaled).to(DEVICE)
    
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        pred_class = int(np.argmax(probs))
        confidence = probs[pred_class] * 100
    
    return DISEASE_MAP[pred_class], confidence, probs

# ────────────────────────────────────────────────────────────────
#                      INTERACTIVE INTERFACE
# ────────────────────────────────────────────────────────────────
def health_assessment_chatbot():
    print("\n" + "═"*60)
    print("      Welcome to Advanced AI Health Risk Assessment")
    print("═"*60)
    print("Please enter your parameters or type 'exit' to quit.\n")
    
    while True:
        try:
            print("─"*70)
            age_str = input("Age (years)                  : ").strip()
            if age_str.lower() in ['exit','quit','q','']:
                print("\nThank you. Stay healthy!\n")
                break
                
            age = float(age_str)
            glucose = float(input("Fasting glucose (mg/dL)      : "))
            sbp = float(input("Systolic BP (mmHg)           : "))
            dbp = float(input("Diastolic BP (mmHg)          : "))
            chol = float(input("Total Cholesterol (mg/dL)    : "))
            hdl = float(input("HDL Cholesterol (mg/dL)      : "))
            tg = float(input("Triglycerides (mg/dL)        : "))
            bmi = float(input("BMI (kg/m²)                  : "))
            
            print("\nAnalyzing health parameters", end="")
            for _ in range(5):
                time.sleep(0.25)
                print(".", end="", flush=True)
            print()
            
            disease, conf, probs = predict_patient(
                age=age, fasting_glucose=glucose, systolic_bp=sbp, diastolic_bp=dbp,
                total_cholesterol=chol, hdl_cholesterol=hdl, triglycerides=tg, bmi=bmi
            )
            
            print(f"\n╔════════════════════════════════════════════════════════════╗")
            print(f"║               HEALTH RISK ASSESSMENT RESULT                ║")
            print(f"╚════════════════════════════════════════════════════════════╝")
            print(f"  Most likely condition : {disease}")
            print(f"  Model confidence      : {conf:.1f}%")
            
            if conf < 75:
                print("\nProbability distribution:")
                for i, p in enumerate(probs):
                    print(f"    {DISEASE_MAP[i]:<22} {p*100:5.1f}%")
            
            print(f"\nRecommendations for {disease}:")
            for i, rec in enumerate(RECOMMENDATIONS[disease], 1):
                print(f"  {i:2d}. {rec}")
            
            print("\n" + "═"*60)
            print("for more knowledge and personalized advice, please consult a healthcare professional.\n")
            print("Always consult a qualified physician.")
            print("═"*60 + "\n")
            
        except ValueError:
            print("Error: Please enter valid numeric values.\n")
        except KeyboardInterrupt:
            print("\nSession terminated.")
            break

if __name__ == "__main__":
    health_assessment_chatbot()