"""
Feature reconstruction for the fraud detection API.
"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
import logging

from api.config import MODELS_DIR

logger = logging.getLogger(__name__)

# Global lookup tables
card_lookup = None
product_lookup = None
p_email_lookup = None
r_email_lookup = None
addr1_lookup = None
addr2_lookup = None
global_stats = None
target_encodings = None
label_encoders = None
feature_names = None

def load_lookup_tables():
    global card_lookup, product_lookup, p_email_lookup, r_email_lookup
    global addr1_lookup, addr2_lookup, global_stats, target_encodings
    global label_encoders, feature_names

    try:
        card_lookup = pd.read_pickle(MODELS_DIR / "card_lookup.pkl")
        logger.info(f"card_lookup loaded: {len(card_lookup):,} cards")

        product_lookup = pd.read_pickle(MODELS_DIR / "product_lookup.pkl")
        logger.info(f"product_lookup loaded: {len(product_lookup):,} products")

        p_email_lookup = pd.read_pickle(MODELS_DIR / "p_email_lookup.pkl")
        logger.info(f"p_email_lookup loaded: {len(p_email_lookup):,} domains")

        r_email_lookup = pd.read_pickle(MODELS_DIR / "r_email_lookup.pkl")
        logger.info(f"r_email_lookup loaded: {len(r_email_lookup):,} domains")

        addr1_lookup = pd.read_pickle(MODELS_DIR / "addr1_lookup.pkl")
        logger.info(f"addr1_lookup loaded: {len(addr1_lookup):,} addresses")

        addr2_lookup = pd.read_pickle(MODELS_DIR / "addr2_lookup.pkl")
        logger.info(f"addr2_lookup loaded: {len(addr2_lookup):,} addresses")

        with open(MODELS_DIR / "global_stats.json", 'r') as f:
            global_stats = json.load(f)
        logger.info("Global stats loaded")

        with open(MODELS_DIR / "target_encodings_v1.json", 'r') as f:
            target_encodings = json.load(f)
        logger.info("Target encodings loaded.")

        with open(MODELS_DIR/ "label_encoders_v1.pkl", "rb") as f:
            label_encoders = pickle.load(f)
        logger.info(f"Label emcoders loaded: {len(label_encoders)} encoders.")

        with open(MODELS_DIR / "feature_names_v1.pkl", 'rb') as f:
            feature_names = pickle.load(f)
        logger.info(f"Feature names loaded: {len(feature_names)} features.")

        all_loaded = all([
            card_lookup is not None,
            product_lookup is not None,
            p_email_lookup is not None,
            r_email_lookup is not None,
            addr1_lookup is not None,
            addr2_lookup is not None,
            global_stats is not None,
            target_encodings is not None,
            label_encoders is not None,
            feature_names is not None
        ])
        
        if all_loaded:
            logger.info("All lookup tables loaded successfully")
            return True
        else:
            logger.warning("Some lookup tables failed to load")
            return False
    except Exception as e:
        logger.error(f"Error loading lookup tables")
        return False

def reconstruct_features(transaction: dict) -> pd.DataFrame:
    features = {}

    # Original transaction features
    features['TransactionDT'] = transaction.get('TransactionDT', 43200)
    features['TransactionAmt'] = transaction.get('TransactionAmt', 0)
    features['ProductCD'] = transaction.get('ProductCD', 'W')
    features['card1'] = transaction.get('card1', np.nan)
    features['card2'] = transaction.get('card2', np.nan)
    features['card3'] = transaction.get('card3', np.nan)
    features['card4'] = transaction.get('card4', np.nan)
    features['card5'] = transaction.get('card5', np.nan)
    features['card6'] = transaction.get('card6', np.nan)
    features['addr1'] = transaction.get('addr1', np.nan)
    features['addr2'] = transaction.get('addr2', np.nan)
    features['dist1'] = transaction.get('dist1', np.nan)
    features['dist2'] = transaction.get('dist2', np.nan)
    features['P_emaildomain'] = transaction.get('P_emaildomain', 'gmail.com')
    features['R_emaildomain'] = transaction.get('R_emaildomain', np.nan)

    for col in feature_names:
        if col not in features:
            features[col] = np.nan

    # Missing value flags
    missing_flags_col = [col.replace('_is_missing', '') for col in feature_names if '_is_missing' in col]

    for col in missing_flags_col:
        flag_name = f"{col}_is_missing"
        if flag_name in feature_names:
            features[flag_name] = 1 if pd.isna(features.get(col, np.nan)) else 0
        
    # Temporal features
    transaction_dt = transaction.get('TransactionDT', 0)

    features['transaction_hour'] = (transaction_dt // 3600) % 24
    features['transaction_day'] = transaction_dt // (3600 * 24)
    features['transaction_weekday'] = (features['transaction_day'] % 7)
    features['is_weekend'] = 1 if features['transaction_weekday'] >=5 else 0
    features['is_night'] = 1 if features['transaction_hour'] < 6 else 0

    hour = features['transaction_hour']
    if hour < 6:
        features['time_period'] = 0
    elif hour < 12:
        features['time_period'] = 1
    elif hour < 18:
        features['time_period'] = 2
    else:
        features['time_period'] = 3
    
    logger.info(f"global_stats keys: {list(global_stats.keys()) if global_stats else 'None'}")
    
    # Velocity features
    features['time_since_last_txn_hrs'] = global_stats.get('time_since_last_global', 24)
    features['txn_count_1h'] = 0
    features['txn_count_24h'] = 0
    features['txn_frequency'] = global_stats['txn_frequency_global']

    # Risk indicators
    features['is_moderate_velocity'] = 0
    features['is_suspicious_time'] = (
        1 if (features['is_night'] == 1 and features['is_weekend'] == 1) else 0
    )

    features['is_rapid_succession'] = 0
    features['is_peak_fraud_hour'] = (
        1 if (2 <= features['transaction_hour'] <= 5) else 0
    )
    features['is_unusual_frequency'] = 0

    # Card aggregations
    card1 = transaction.get('card1', None)

    if card1 is not None and card1 in card_lookup.index:
        card_row = card_lookup.loc[card1]
        features['card_amt_mean'] = card_row['card_amt_mean']
        features['card_amt_std'] = card_row['card_amt_std']
        features['card_amt_min'] = card_row['card_amt_min']
        features['card_amt_max'] = card_row['card_amt_max']
        features['card_amt_median'] = card_row['card_amt_median']
        features['card_txn_count'] = card_row['card_txn_count']
        features['card_fraud_rate'] = card_row['card_fraud_rate']
    else:
        # Unknown card - use global defaults
        features['card_amt_mean'] = global_stats['card_amt_mean_global']
        features['card_amt_std'] = global_stats['card_amt_std_global']
        features['card_amt_min'] = global_stats['median_amount']
        features['card_amt_max'] = global_stats['median_amount']
        features['card_amt_median'] = global_stats['median_amount']
        features['card_txn_count'] = 1
        features['card_fraud_rate'] = global_stats['card_fraud_rate_global']

    # Derived card features
    features['card_amt_range'] = features['card_amt_max'] - features['card_amt_min']
    features['card_deviation'] = (
        features['TransactionAmt'] - features['card_amt_mean']
    )
    features['amt_ratio_card_mean'] = (
        features['TransactionAmt'] / (features['card_amt_mean'] + 0.01)
    )
    features['is_outside_card_range'] = (
        1 if (features['TransactionAmt'] < features['card_amt_min'] or
              features['TransactionAmt'] > features['card_amt_max']) else 0
    )

    # Product aggregations
    product = transaction.get('ProductCD', 'W')

    if product in product_lookup.index:
        prod_row = product_lookup.loc[product]
        features['product_amt_mean'] = prod_row['product_amt_mean']
        features['product_amt_std'] = prod_row['product_amt_std']
        features['product_amt_median'] = prod_row['product_amt_median']
        features['product_txn_count'] = prod_row['product_txn_count']
        features['product_fraud_rate'] = prod_row['product_fraud_rate']
    else:
        features['product_amt_mean'] = global_stats['median_amount']
        features['product_amt_std'] = 50.0
        features['product_amt_median'] = global_stats['median_amount']
        features['product_txn_count'] = 1000
        features['product_fraud_rate'] = global_stats['product_fraud_rate_global']

    features['product_deviation'] = (
        features['TransactionAmt'] - features['product_amt_mean']
    )
    features['amt_ratio_product_mean'] = (
        features['TransactionAmt'] / (features['product_amt_mean'] + 0.01)
    )

    # Email domain aggregations
    p_email = transaction.get('P_emaildomain', 'gmail.com')
    r_email = transaction.get('R_emaildomain', None)

    if p_email and p_email in p_email_lookup.index:
        p_row = p_email_lookup.loc[p_email]
        features['P_email_amt_mean'] = p_row['P_email_amt_mean']
        features['P_email_txn_count'] = p_row['P_email_txn_count']
        features['P_email_fraud_rate'] = p_row['P_email_fraud_rate']
    else:
        features['P_email_amt_mean'] = global_stats['median_amount']
        features['P_email_txn_count'] = 100
        features['P_email_fraud_rate'] = global_stats['p_email_fraud_rate_global']

    if r_email and r_email in r_email_lookup.index:
        r_row = r_email_lookup.loc[r_email]
        features['R_email_amt_mean'] = r_row['R_email_amt_mean']
        features['R_email_txn_count'] = r_row['R_email_txn_count']
        features['R_email_fraud_rate'] = r_row['R_email_fraud_rate']
    else:
        features['R_email_amt_mean'] = global_stats['median_amount']
        features['R_email_txn_count'] = 100
        features['R_email_fraud_rate'] = global_stats['r_email_fraud_rate_global']
    
    # Address aggregations
    addr1 = transaction.get('addr1', None)
    addr2 = transaction.get('addr2', None)

    if addr1 and addr1 in addr1_lookup.index:
        a1_row = addr1_lookup.loc[addr1]
        features['addr1_amt_mean'] = a1_row['addr1_amt_mean']
        features['addr1_txn_count'] = a1_row['addr1_txn_count']
        features['addr1_fraud_rate'] = a1_row['addr1_fraud_rate']
    else:
        features['addr1_amt_mean'] = global_stats['median_amount']
        features['addr1_txn_count'] = 100
        features['addr1_fraud_rate'] = global_stats['addr1_fraud_rate_global']

    if addr2 and addr2 in addr2_lookup.index:
        a2_row = addr2_lookup.loc[addr2]
        features['addr2_amt_mean'] = a2_row['addr2_amt_mean']
        features['addr2_txn_count'] = a2_row['addr2_txn_count']
        features['addr2_fraud_rate'] = a2_row['addr2_fraud_rate']
    else:
        features['addr2_amt_mean'] = global_stats['median_amount']
        features['addr2_txn_count'] = 100
        features['addr2_fraud_rate'] = global_stats['addr2_fraud_rate_global']

    numeric_fields = [
    'card1', 'card2', 'card3', 'card5',
    'addr1', 'addr2', 'dist1', 'dist2'
    ]

    for field in numeric_fields:
        val = features.get(field, np.nan)
        try:
            features[field] = float(val) if val is not None else np.nan
        except (TypeError, ValueError):
            features[field] = np.nan

    # Build a dataframe that aligns with the model
    df  = pd.DataFrame([features])

    for col in df.columns:
        if col not in [c for c in df.columns if '_is_missing' in c]:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception:
                pass

    # Encode categorical columns
    for col, le in label_encoders.items():
        if col in df.columns:
            val = str(df[col].iloc[0])
            if val in le.classes_:
                df[col] = le.transform([val])[0]
            else:
                df[col] = le.transform([le.classes_[0]])[0]

    # Align to exact feature model expects
    for col in feature_names:
        if col not in df.columns:
            df[col] = -999
    
    df[feature_names]
    df = df.fillna(-999)

    return df