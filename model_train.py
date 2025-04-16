import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import pickle

# Load data
df = pd.read_csv("StudentsPerformance.csv")

# Create average score and result
df["avg_score"] = df[["math score", "reading score", "writing score"]].mean(axis=1)
df["result"] = df["avg_score"].apply(lambda x: 1 if x >= 50 else 0)

# Drop original scores
df.drop(["math score", "reading score", "writing score", "avg_score"], axis=1, inplace=True)

# Encode categorical columns
label_encoders = {}
for col in df.columns[:-1]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Check label distribution
print(df["result"].value_counts())

# Split
X = df.drop("result", axis=1)
y = df["result"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train SVM with class weight
model = SVC(probability=True, class_weight='balanced')
model.fit(X_train, y_train)

# Save model and encoders
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)
