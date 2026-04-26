import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ── Column definitions ────────────────────────────────────────────────────────
columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
    'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
    'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login',
    'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
    'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
    'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'label', 'difficulty'
]

# ── Load & clean ──────────────────────────────────────────────────────────────
train_data = pd.read_csv("Dataset/KDDTrain+.TXT", names=columns)
test_data  = pd.read_csv("Dataset/KDDTest+.TXT",  names=columns)

train_data.drop('difficulty', axis=1, inplace=True)
test_data.drop('difficulty',  axis=1, inplace=True)

# ── Binary labels ─────────────────────────────────────────────────────────────
train_data['label'] = (train_data['label'] != 'normal').astype(int)
test_data['label']  = (test_data['label']  != 'normal').astype(int)

# ── Encode categoricals (safe against unseen test labels) ─────────────────────
def safe_label_encode(train_col, test_col):
    le = LabelEncoder()
    le.fit(train_col)
    known = set(le.classes_)
    # Unseen values fall back to the most frequent training value
    fallback = train_col.mode()[0]
    test_col = test_col.apply(lambda x: x if x in known else fallback)
    return le.transform(train_col), le.transform(test_col)

cat_cols = ['protocol_type', 'service', 'flag']
for col in cat_cols:
    train_data[col], test_data[col] = safe_label_encode(
        train_data[col], test_data[col]
    )

# ── Features / labels ─────────────────────────────────────────────────────────
X_train = train_data.drop('label', axis=1)
y_train = train_data['label']
X_test  = test_data.drop('label', axis=1)
y_test  = test_data['label']

# ── Model ─────────────────────────────────────────────────────────────────────
import streamlit as st

@st.cache_resource
def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced',
        max_depth=20,
        min_samples_leaf=5,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

model = train_model(X_train, y_train)

# # ── Default Evaluation ─────────────────────────────────────────────────────────
# y_pred = model.predict(X_test)

# print("=== Default Threshold (0.5) ===")
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
# print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Normal', 'Attack']))

# ── Threshold Tuning ───────────────────────────────────────────────────────────
from sklearn.metrics import precision_recall_curve

y_proba = model.predict_proba(X_test)[:, 1]

precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

# Compute F1 scores safely
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)

best_index = f1_scores[:-1].argmax()
best_thresh = thresholds[best_index]

print(f"\nOptimal Threshold (F1): {best_thresh:.3f}")

# ── Evaluation with tuned threshold ────────────────────────────────────────────
y_pred_tuned = (y_proba >= best_thresh).astype(int)
# print("\n=== Tuned Threshold Results ===")
# print("Accuracy:", accuracy_score(y_test, y_pred_tuned))
# print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_tuned))
# print("\nClassification Report:\n", classification_report(y_test, y_pred_tuned, target_names=['Normal', 'Attack']))

# ── Feature importance ────────────────────────────────────────────────────────
# feat_importance = pd.Series(model.feature_importances_, index=X_train.columns)
# print("\nTop 10 Most Important Features:")
# print(feat_importance.sort_values(ascending=False).head(10))

# # ── Real-time simulation ─────────────────────────────────────────────────────
# import time

# try:
#     for i in range(10):
#         sample = X_test.sample(1)
        
#         prob = model.predict_proba(sample)[:,1][0]
#         pred = 1 if prob >= best_thresh else 0

#         if prob > 0.7:
#             msg = "[HIGH RISK]"
#         elif prob > 0.4:
#             msg = "[MEDIUM RISK]"
#         else:
#             msg = "[LOW RISK]"

#         if pred == 1:
#             print(f"{msg} Attack detected! Probability: {prob:.2f}")
#         else:
#             print(f"[SAFE] Normal traffic. Probability: {prob:.2f}")
        
#         time.sleep(10)

# except KeyboardInterrupt:
#     print("\nSimulation stopped cleanly.")

st.title("🚨 Network Intrusion Detection System")

st.write("Click the button to simulate incoming network traffic.")

if st.button("Simulate Traffic"):
    sample = X_test.sample(1)
    prob = model.predict_proba(sample)[:,1][0]
    pred = 1 if prob >= best_thresh else 0

    if prob > 0.7:
        risk = "HIGH RISK"
    elif prob > 0.4:
        risk = "MEDIUM RISK"
    else:
        risk = "LOW RISK"

    if pred == 1:
        st.error(f"{risk} 🚨 Attack Detected! Probability: {prob:.2f}")
    else:
        st.success(f"SAFE ✅ Normal Traffic. Probability: {prob:.2f}")