
# data_verification.py
import pandas as pd

def verify_data_splits():
    print("📊 VERIFYING DATA SPLITS")
    print("=" * 50)
    
    # Check training data
    train_df = pd.read_csv("bitcoin_train.csv")
    train_df['Open time'] = pd.to_datetime(train_df['Open time'])
    
    # Check test data  
    test_df = pd.read_csv("bitcoin_test.csv")
    test_df['Open time'] = pd.to_datetime(test_df['Open time'])
    
    print("✅ TRAINING DATA:")
    print(f"   Rows: {len(train_df)}")
    print(f"   Date Range: {train_df['Open time'].min()} to {train_df['Open time'].max()}")
    
    print("🧪 TEST DATA:")
    print(f"   Rows: {len(test_df)}") 
    print(f"   Date Range: {test_df['Open time'].min()} to {test_df['Open time'].max()}")
    
    # Check for overlaps
    train_max = train_df['Open time'].max()
    test_min = test_df['Open time'].min()
    
    print("\n🔍 OVERLAP CHECK:")
    if train_max < test_min:
        print("   ✅ NO OVERLAP - Training ends before test begins")
        print(f"   Gap: {test_min - train_max} days between sets")
    else:
        print("   ❌ DATA LEAKAGE DETECTED! Overlap found!")
        print(f"   Training ends: {train_max}")
        print(f"   Test starts: {test_min}")
    
    # Check if test data is contained in training
    test_dates_in_train = test_df['Open time'].isin(train_df['Open time']).any()
    print(f"   Test dates in training: {test_dates_in_train}")

if __name__ == "__main__":
    verify_data_splits()