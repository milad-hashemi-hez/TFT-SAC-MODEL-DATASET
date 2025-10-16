
# data_splitter.py

import pandas as pd
from datetime import datetime, timedelta

def split_and_save_bitcoin_data(csv_path="bitcoin_data.csv", test_days=90):
    """
    Split Bitcoin dataset into train and test files.
    Training: All data except last X days
    Testing: Last X days
    
    Args:
        csv_path (str): Path to Bitcoin CSV file
        test_days (int): Number of days for testing (default: 90 = 3 months)
    
    Returns:
        tuple: (train_file_path, test_file_path)
    """
    
    print("🔄 Loading Bitcoin data...")
    df = pd.read_csv(csv_path)
    df['Open time'] = pd.to_datetime(df['Open time'])
    df = df.sort_values('Open time').reset_index(drop=True)
    
    print(f"📊 Loaded {len(df)} rows")
    print(f"📅 Data range: {df['Open time'].min().date()} to {df['Open time'].max().date()}")
    
    # Calculate split point (last X days for testing)
    split_date = df['Open time'].max() - timedelta(days=test_days)
    
    # Split data
    train_df = df[df['Open time'] < split_date].copy()
    test_df = df[df['Open time'] >= split_date].copy()
    
    # Save to separate files
    train_file = "bitcoin_train.csv"
    test_file = "bitcoin_test.csv"
    
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)
    
    print(f"\n✅ SPLIT COMPLETE:")
    print(f"   TRAIN: {len(train_df)} rows → {train_file}")
    print(f"   TEST:  {len(test_df)} rows → {test_file}")
    print(f"   Train dates: {train_df['Open time'].min().date()} to {train_df['Open time'].max().date()}")
    print(f"   Test dates:  {test_df['Open time'].min().date()} to {test_df['Open time'].max().date()}")
    
    return train_file, test_file

def load_training_data():
    """Load the training data from saved file"""
    return pd.read_csv("bitcoin_train.csv")

def load_test_data():
    """Load the test data from saved file"""
    return pd.read_csv("bitcoin_test.csv")

# Run the split automatically when this module is executed
if __name__ == "__main__":
    train_path, test_path = split_and_save_bitcoin_data("bitcoin_data.csv", test_days=90)
    print(f"\n🎯 Ready to use:")
    print(f"   from data_splitter import load_training_data, load_test_data")
    print(f"   train_df = load_training_data()")
    print(f"   test_df = load_test_data()")