import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import logging

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def retrain_model():
    logger.info("Retraining process shuru ho raha hai...")
    
    # 1. Naya data load karein (Jo drift ke baad collect hua)
    # Aap yahan apna naya/current data load karein
    train_df = pd.read_csv('data/train_data.csv') 
    
    # 2. Features aur Target separate karein
    X = train_df.drop('target', axis=1) # Apna target column name check kar lein
    y = train_df['target']
    
    # 3. Model Train karein
    model = RandomForestRegressor()
    model.fit(X, y)
    
    # 4. MLflow mein naya model register karein
    mlflow.set_tracking_uri("http://localhost:5000")
    with mlflow.start_run():
        mlflow.sklearn.log_model(model, "model")
        logger.info("Naya model MLflow mein register ho gaya hai!")

if __name__ == "__main__":
    retrain_model()