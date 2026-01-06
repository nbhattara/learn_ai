import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import warnings
import time

warnings.filterwarnings("ignore", category=UserWarning)

# ────────────────────────────────────────────────────────────────
#                     CONFIGURATION
# ────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.20
BATCH_SIZE = 16
EPOCHS = 120
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DISEASE_MAP = {
    0: "Healthy",
    1: "Diabetes",
    2: "Heart Disease",
    3: "Hypertension"
}

RECOMMENDATIONS = {
    "Healthy": [
        "Maintain balanced nutrition with adequate vegetables and fruits",
        "Engage in moderate physical activity at least 150 minutes per week",
        "Ensure 7–9 hours of quality sleep per night",
        "Schedule annual preventive health examinations",
        "Avoid tobacco and limit alcohol consumption"
    ],
    "Diabetes": [
        "Significantly reduce intake of refined sugars and processed carbohydrates",
        "Monitor blood glucose levels regularly",
        "Follow a structured, low-glycemic index meal plan",
        "Perform regular physical activity under medical guidance",
        "Maintain regular consultations with an endocrinologist",
        "Aim for healthy body weight and waist circumference"
    ],
    "Heart Disease": [
        "Adopt a Mediterranean-style or heart-healthy dietary pattern",
        "Completely eliminate tobacco use and avoid secondhand smoke",
        "Participate in medically supervised cardiovascular exercise",
        "Schedule periodic cardiac evaluations (ECG, echo, stress test)",
        "Maintain strict control of blood pressure, lipids, and glucose",
        "Practice evidence-based stress reduction techniques"
    ],
    "Hypertension": [
        "Restrict sodium intake to less than 1500–2300 mg per day",
        "Follow the DASH dietary pattern consistently",
        "Engage in regular aerobic exercise (most days of the week)",
        "Perform home blood pressure monitoring with validated device",
        "Limit alcohol to recommended guidelines",
        "Implement structured stress management strategies",
        "Maintain regular follow-up with a cardiologist"
    ]
}

# ────────────────────────────────────────────────────────────────
#                   EXTENDED DEMONSTRATION DATA
# ────────────────────────────────────────────────────────────────
data = {
    'age': [24,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,59,61,63,65,67,69,71,73],
    'glucose': [82,86,89,93,97,102,108,114,122,130,138,146,155,164,172,181,189,198,208,219,230,242,255,268,282],
    'systolic_bp': [108,112,116,119,122,126,130,135,139,144,149,154,159,164,169,174,180,186,192,198,205,212,218,225,232],
    'cholesterol': [138,145,152,158,165,172,180,188,196,204,212,220,228,236,245,254,263,272,282,292,302,313,324,336,348],
    'bmi': [19.8,21.1,22.4,23.6,24.8,26.1,27.3,28.5,29.7,30.9,32.2,33.4,34.6,35.8,37.1,38.3,39.6,40.9,42.2,43.5,44.8,46.2,47.6,49.0,50.4],
    'disease': [0,0,0,0,0,0,1,1,1,2,2,2,3,3,3,3,2,2,1,1,3,3,3,2,1]
}

df = pd.DataFrame(data)

# ────────────────────────────────────────────────────────────────
#                     DATA PREPARATION
# ────────────────────────────────────────────────────────────────
X = df.drop('disease', axis=1).values.astype(np.float32)
y = df['disease'].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# ────────────────────────────────────────────────────────────────
#                        PYTORCH DATASET
# ────────────────────────────────────────────────────────────────
class PatientDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).long()
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = PatientDataset(X_train, y_train)
test_dataset  = PatientDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

# ────────────────────────────────────────────────────────────────
#                        NEURAL NETWORK
# ────────────────────────────────────────────────────────────────
class HealthClassifier(nn.Module):
    def __init__(self, input_size=4, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.35),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.25),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.15),
            
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)

model = HealthClassifier(input_size=X.shape[1]).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=12)

# ────────────────────────────────────────────────────────────────
#                        TRAINING LOOP
# ────────────────────────────────────────────────────────────────
print(f"Training neural network on {DEVICE}...\n")

best_acc = 0.0
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()
    
    train_acc = correct / total
    train_loss /= len(train_loader)
    
    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            val_total += y_batch.size(0)
            val_correct += (predicted == y_batch).sum().item()
    
    val_acc = val_correct / val_total
    
    scheduler.step(train_loss)
    
    if (epoch + 1) % 20 == 0 or epoch == EPOCHS-1:
        print(f"Epoch [{epoch+1:3d}/{EPOCHS}] | "
              f"Train Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.3%} | "
              f"Val Acc: {val_acc:.3%}")
    
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_health_model.pt")

print(f"\nTraining completed. Best validation accuracy: {best_acc:.3%}\n")

# Load best model
model.load_state_dict(torch.load("best_health_model.pt", map_location=DEVICE))
model.eval()

# ────────────────────────────────────────────────────────────────
#                      CHATBOT INTERFACE
# ────────────────────────────────────────────────────────────────
def predict_patient(age, glucose, systolic_bp, cholesterol, bmi):
    features = np.array([[age, glucose, systolic_bp, cholesterol, bmi]], dtype=np.float32)
    features_scaled = scaler.transform(features)
    tensor = torch.from_numpy(features_scaled).to(DEVICE)
    
    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        predicted_class = np.argmax(probabilities)
        confidence = probabilities[predicted_class] * 100
    
    disease = DISEASE_MAP[predicted_class]
    return disease, confidence, probabilities

def print_recommendations(disease, confidence):
    print(f"\n╔════════════════════════════════════════════════════╗")
    print(f"║          MEDICAL CONDITION ASSESSMENT              ║")
    print(f"╚════════════════════════════════════════════════════╝")
    print(f"Estimated condition    : {disease}")
    print(f"Model confidence       : {confidence:.1f}%")
    print(f"\nRecommended actions:")
    
    for i, action in enumerate(RECOMMENDATIONS[disease], 1):
        print(f"  {i}. {action}")
    
    print("\n⚠️  IMPORTANT NOTICE")
    print("This is an educational demonstration system only.")
    print("It is NOT a substitute for professional medical advice,")
    print("diagnosis, or treatment. Always consult a qualified physician.")
    print("═══════════════════════════════════════════════════════\n")

# ────────────────────────────────────────────────────────────────
#                      INTERACTIVE CHATBOT
# ────────────────────────────────────────────────────────────────
def medical_chatbot():
    print("┌────────────────────────────────────────────────────┐")
    print("│           Welcome to Health Assessment Demo        │")
    print("│   (Educational neural network prototype - 2025)    │")
    print("└────────────────────────────────────────────────────┘")
    print("Please provide your health parameters or type 'exit' to quit.\n")
    
    while True:
        try:
            print("─" * 60)
            age = input("Age (years): ").strip()
            if age.lower() in ['exit', 'quit', 'q']:
                print("\nThank you for using the demo. Take care of your health.")
                break
                
            glucose = float(input("Fasting glucose (mg/dL): "))
            bp = float(input("Systolic blood pressure (mmHg): "))
            chol = float(input("Total cholesterol (mg/dL): "))
            bmi = float(input("BMI (kg/m²): "))
            
            print("\nAnalyzing", end="")
            for _ in range(5):
                time.sleep(0.4)
                print(".", end="", flush=True)
            print()
            
            disease, confidence, probs = predict_patient(
                float(age), glucose, bp, chol, bmi
            )
            
            print_recommendations(disease, confidence)
            
            # Optional: show probability breakdown
            if confidence < 65:
                print("Detailed probabilities:")
                for i, p in enumerate(probs):
                    print(f"  • {DISEASE_MAP[i]:<15} {p*100:5.1f}%")
                print()
            
        except ValueError:
            print("Error: Please enter valid numeric values.\n")
        except KeyboardInterrupt:
            print("\nSession terminated by user.")
            break

# Start the chatbot
if __name__ == "__main__":
    medical_chatbot()