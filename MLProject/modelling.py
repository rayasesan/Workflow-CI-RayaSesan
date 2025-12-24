# modelling.py untuk CI
import mlflow
import mlflow.sklearn
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

# SET TRACKING URI KE FOLDER LOKAL
mlflow.set_tracking_uri("file://./mlruns")  

def load_data():
    data_dir = "titanic_preprocessed"
    X = np.load(os.path.join(data_dir, "X_scaled.npy"))
    y = np.load(os.path.join(data_dir, "y.npy"))
    return X, y

def main():
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
        
        mlflow.log_param("n_estimators", 50)
        mlflow.log_metric("accuracy", accuracy)
        
        print(f"Model trained. Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
