
"""
demo_agent.py - Simple Goods Management AI Agent demo script.

This script demonstrates:
- loading a sample goods CSV
- simple feature engineering
- training a RandomForestRegressor to predict sold quantity next period
- producing simple recommendations (restock if predicted demand > current stock threshold)
- saving predictions to output CSV

Usage:
    python demo_agent.py --input sample_goods.csv --output predictions.csv
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

def load_data(path):
    df = pd.read_csv(path, parse_dates=['date'])
    return df

def preprocess(df):
    # Basic features: product_id, category encoded, previous sold_qty, stock
    df = df.sort_values(['product_id','date'])
    # lag feature: previous sold_qty per product
    df['prev_sold'] = df.groupby('product_id')['sold_qty'].shift(1).fillna(method='bfill')
    # day of year to capture seasonality
    df['dayofyear'] = df['date'].dt.dayofyear
    # encode category simple mapping
    cat_map = {c:i for i,c in enumerate(df['category'].unique())}
    df['cat_enc'] = df['category'].map(cat_map)
    return df, cat_map

def train_model(df):
    # We'll predict next period's sold_qty using current row features
    df = df.copy()
    df['target'] = df.groupby('product_id')['sold_qty'].shift(-1)  # next sold_qty
    df = df.dropna(subset=['target'])
    features = ['prev_sold','stock','dayofyear','cat_enc']
    X = df[features]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("MAE on test:", mean_absolute_error(y_test, preds))
    return model, features

def make_predictions(model, features, df, output_path):
    df_pred = df.copy()
    # For simplicity, predict next period sold_qty for current rows (use prev_sold computed earlier)
    X = df_pred[features].fillna(0)
    df_pred['pred_next_sold'] = model.predict(X)
    # Recommendation: restock if predicted next sold_qty > current stock * 0.6
    df_pred['recommendation'] = np.where(df_pred['pred_next_sold'] > df_pred['stock']*0.6, 'Restock', 'OK')
    df_pred.to_csv(output_path, index=False)
    print(f"Predictions and recommendations saved to: {output_path}")
    return df_pred

def save_model(model, path):
    joblib.dump(model, path)
    print(f"Model saved to: {path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="sample_goods.csv", help="Input CSV file path")
    parser.add_argument("--output", default="predictions.csv", help="Output CSV file path")
    parser.add_argument("--model", default="rf_model.joblib", help="Model save path")
    args = parser.parse_args()

    df = load_data(args.input)
    df, cat_map = preprocess(df)
    model, features = train_model(df)
    save_model(model, args.model)
    out = make_predictions(model, features, df, args.output)
    # Show sample recommendations
    print(out[['product_id','product_name','date','stock','pred_next_sold','recommendation']].tail(10))

if __name__ == "__main__":
    main()
