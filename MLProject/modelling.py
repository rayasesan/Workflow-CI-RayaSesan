import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

# SET PATH ABSOLUT KE ./mlruns (folder di MLProject)
current_dir = os.path.dirname(os.path.abspath(__file__))
mlruns_path = os.path.join(current_dir, "mlruns")
mlflow.set_tracking_uri(f"file://{mlruns_path}")

def load_data():
    data_dir = "titanic_preprocessed"
    X = np.load(os.path.join(data_dir, "X_scaled.npy"))
    y = np.load(os.path.join(data_dir, "y.npy"))
    return X, y

def main():
    # BUAT FOLDER mlruns JIKA BELUM ADA
    os.makedirs(mlruns_path, exist_ok=True)
    
    mlflow.sklearn.autolog()
    
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model trained. Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
