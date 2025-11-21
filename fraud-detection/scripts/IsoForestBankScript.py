import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from IsoForestBankData import BankTransactionIsoForestModel


def print_report(results_df: pd.DataFrame, top_n: int = 10):
    total_samples = len(results_df)
    n_anomalies = (results_df['anomaly_label'] == -1).sum()
    anomaly_rate = n_anomalies / total_samples
    
    print("\n" + "="*80)
    print("ANOMALY DETECTION REPORT")
    print("="*80)
    print(f"Total transactions: {total_samples:,}")
    print(f"Anomalies detected: {n_anomalies:,} ({anomaly_rate:.2%})")
    print(f"Normal transactions: {total_samples - n_anomalies:,} ({1 - anomaly_rate:.2%})")
    print("\n" + "-"*80)
    
    print(f"\nTop {top_n} Most Suspicious Transactions (lowest anomaly scores):")
    print("-"*80)
    
    top_anomalies = results_df.nsmallest(top_n, 'anomaly_score')
    
    display_cols = ['TransactionID', 'AccountID', 'TransactionAmount', 
                    'TransactionDate', 'TransactionType', 'Channel', 
                    'AccountBalance', 'anomaly_score', 'anomaly_label']
    available_cols = [col for col in display_cols if col in top_anomalies.columns]
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 30)
    
    print(top_anomalies[available_cols].to_string(index=False))
    
    print("\n" + "="*80)
    
    if n_anomalies > 0:
        print("\nAnomaly Statistics:")
        print("-"*80)
        anomaly_df = results_df[results_df['anomaly_label'] == -1]
        print(f"Average transaction amount (anomalies): ${anomaly_df['TransactionAmount'].mean():.2f}")
        print(f"Average transaction amount (normal): ${results_df[results_df['anomaly_label'] == 1]['TransactionAmount'].mean():.2f}")
        print(f"Average account balance (anomalies): ${anomaly_df['AccountBalance'].mean():.2f}")
        print(f"Average account balance (normal): ${results_df[results_df['anomaly_label'] == 1]['AccountBalance'].mean():.2f}")


def main():
    parser = argparse.ArgumentParser(
        description='Run Isolation Forest anomaly detection on bank transaction data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Optional arguments
    parser.add_argument(
        '--data-path',
        type=str,
        default='bank_transactions_data_2.csv',
        help='Path to the bank transaction CSV file (default: bank_transactions_data_2.csv)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Optional: Path to save results CSV file'
    )
    
    args = parser.parse_args()
    
    # Set default values for removed arguments
    contamination = getattr(args, 'contamination', 'auto')
    n_estimators = getattr(args, 'n_estimators', 200)
    random_state = getattr(args, 'random_state', 42)
    top_n = getattr(args, 'top_n', 10)
    
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"Error: Data file not found: {args.data_path}")
        return
    
    print("="*80)
    print("Isolation Forest Anomaly Detection")
    print("="*80)
    print(f"Data file: {args.data_path}")
    print(f"Contamination: {contamination}")
    print(f"Number of estimators: {n_estimators}")
    print(f"Random state: {random_state}")
    print("="*80)
    
    print("\nLoading data...")
    try:
        df = pd.read_csv(args.data_path)
        print(f"Loaded {len(df):,} transactions")
        print(f"Columns: {', '.join(df.columns.tolist())}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    print("\nInitializing model...")
    model = BankTransactionIsoForestModel(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state
        )
    print("Model initialized")

    
    
    print("\nTraining model...")
    model.fit(df)
    
    print("\nGenerating predictions...")
    results = model.predict(df)

    
    if args.output:
        print(f"\nSaving results to {args.output}...")
        try:
            results.to_csv(args.output, index=False)
            print("Results saved successfully")
        except Exception as e:
            print(f"Error saving results: {e}")
    
    print_report(results, top_n=top_n)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
