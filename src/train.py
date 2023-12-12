from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, fbeta_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import argparse
import joblib

# choose a multilingual text embedder
embedder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('--train', required=True, help='Path to the training parquet file')
parser.add_argument('--test', required=True, help='Path to the testing parquet file')

args = parser.parse_args()

train = pd.read_parquet(args.train)
test = pd.read_parquet(args.test)

# Create embeddings
train["embeddings"] = train["text"].apply(embedder.encode)
test["embeddings"] = test["text"].apply(embedder.encode)


# Convert labels to numerical values using LabelEncoder
label_encoder = LabelEncoder()
train['label'] = label_encoder.fit_transform(train['label'])
test['label'] = label_encoder.transform(test['label'])

# Convert embeddings to NumPy arrays
X_train = np.vstack(train['embeddings'].to_numpy())
y_train = train['label'].to_numpy()

X_test = np.vstack(test['embeddings'].to_numpy())
y_test = test['label'].to_numpy()

model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f'Validation Accuracy: {accuracy * 100:.2f}%')
# Calculate F1 score
f1 = f1_score(y_test, predictions, average="weighted")
print(f'F1 Score: {f1 * 100:.2f}%')
# Calculate F2 score
f2 = fbeta_score(y_test, predictions, beta=2, average="weighted")
print(f'F2 Score: {f2 * 100:.2f}%')

# Save the label encoder and trained model
joblib.dump(label_encoder, 'models/label_encoder.joblib')
joblib.dump(model, 'models/logistic_regression_model.joblib')
