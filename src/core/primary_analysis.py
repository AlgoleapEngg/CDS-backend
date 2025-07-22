import os
import logging
import pandas as pd
import numpy as np
from scipy.stats import zscore, kurtosis, skew

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load CSV or Excel into DataFrame, with file existence and format checks.
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: '{file_path}'")

    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == '.csv':
            df = pd.read_csv(file_path)
        elif ext in ('.xlsx', '.xls'):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format. Use .csv, .xlsx or .xls")
    except Exception as e:
        logger.error(f"Failed to load file '{file_path}': {e}")
        raise
    return df

def analyze_spreadsheet(file_path: str) -> dict:
    """
    Master orchestration with structured error handling per step.
    Returns:
        {
            "success": bool,
            "errors": [str],
            "analysis": {...}
        }
    """
    result = {"success": False, "errors": [], "analysis": {}}

    # Step 1: Load data
    try:
        df = load_data(file_path)
    except Exception as e:
        result["errors"].append(str(e))
        return result

    # Step 2: Validate non-empty
    if df.empty:
        msg = "The dataset is empty."
        logger.warning(msg)
        result["errors"].append(msg)
        return result

    # Core analysis steps
    steps = {
        "dataset_overview": get_dataset_overview,
        "data_quality": check_data_quality,
        "value_distributions": analyze_value_distributions,
        "cardinality": check_cardinality,
        "outliers": detect_outliers,
        "correlations": analyze_correlations,
        "datetime_analysis": check_datetime_features,
        "id_detection": detect_ids_and_indexes,
        "target_inference": infer_target_column,
        "column_recommendations": generate_column_recommendations
    }

    for name, func in steps.items():
        try:
            result["analysis"][name] = func(df)
        except Exception as e:
            err = f"{name} failed: {e}"
            logger.error(err)
            result["errors"].append(err)
            result["analysis"][name] = None

    # Feature-type–based analysis
    try:
        types = detect_feature_types(df)
        feature_based = {}
        for col, ftype in types.items():
            if ftype == 'continuous':
                feature_based[col] = analyze_continuous(df, col)
            elif ftype == 'discrete':
                feature_based[col] = analyze_discrete(df, col)
            elif ftype == 'categorical':
                feature_based[col] = analyze_categorical(df, col)
            elif ftype == 'text':
                feature_based[col] = analyze_text(df, col)
            elif ftype == 'boolean':
                feature_based[col] = analyze_boolean(df, col)
            elif ftype == 'datetime':
                feature_based[col] = analyze_datetime(df, col)
            else:
                feature_based[col] = {'note': f'no analysis for type {ftype}'}
        result["analysis"]["feature_based_analysis"] = {
            "feature_types": types,
            "details": feature_based
        }
    except Exception as e:
        err = f"feature_based_analysis failed: {e}"
        logger.error(err)
        result["errors"].append(err)
        result["analysis"]["feature_based_analysis"] = None

    result["success"] = True
    return result

# ——— Helper functions below ———

def get_dataset_overview(df: pd.DataFrame) -> dict:
    return {
        "num_rows": int(len(df)),
        "num_columns": int(len(df.columns)),
        "column_names": df.columns.tolist(),
        "data_types": df.dtypes.astype(str).to_dict(),
        "memory_usage_mb": float(df.memory_usage(deep=True).sum() / 1024 ** 2)
    }

def check_data_quality(df: pd.DataFrame) -> dict:
    return {
        "missing_values": df.isnull().sum().astype(int).to_dict(),
        "missing_value_percentage": (df.isnull().mean() * 100).round(2).to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "constant_columns": [col for col in df.columns if df[col].nunique(dropna=False) == 1],
        "mixed_type_columns": [col for col in df.columns if df[col].dropna().apply(type).nunique() > 1]
    }

def analyze_value_distributions(df: pd.DataFrame) -> dict:
    distributions = {}
    for col in df.columns:
        try:
            if pd.api.types.is_bool_dtype(df[col]):
                series = df[col]
                distributions[col] = {
                    "true_count": int(series.sum()),
                    "false_count": int((~series).sum())
                }
            elif pd.api.types.is_numeric_dtype(df[col]):
                series = pd.to_numeric(df[col], errors='coerce').dropna()
                if len(series) < 2:
                    continue
                distributions[col] = {
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "mean": float(series.mean()),
                    "median": float(series.median()),
                    "std": float(series.std()),
                    "skew": float(skew(series)),
                    "kurtosis": float(kurtosis(series))
                }
            else:
                series = df[col].dropna().astype(str)
                distributions[col] = {
                    "unique_count": int(series.nunique()),
                    "top_values": series.value_counts().head(5).to_dict()
                }
        except Exception:
            distributions[col] = {"error": "Failed to analyze"}
    return distributions

def check_cardinality(df: pd.DataFrame) -> dict:
    return {col: int(df[col].nunique(dropna=True)) for col in df.columns}

def detect_outliers(df: pd.DataFrame) -> dict:
    outliers = {}
    for col in df.select_dtypes(include=np.number).columns:
        try:
            if pd.api.types.is_bool_dtype(df[col]):
                continue
            series = df[col].dropna()
            if len(series) < 2:
                continue
            z_scores = zscore(series)
            outliers[col] = int((np.abs(z_scores) > 3).sum())
        except Exception:
            outliers[col] = "error"
    return outliers

def analyze_correlations(df: pd.DataFrame) -> dict:
    numeric_df = df.select_dtypes(include=np.number).drop(
        columns=[c for c in df.select_dtypes(include=[bool]).columns],
        errors='ignore'
    )
    if numeric_df.shape[1] < 2:
        return {}
    return numeric_df.corr().round(2).replace({np.nan: None}).to_dict()

def check_datetime_features(df: pd.DataFrame) -> dict:
    datetime_analysis = {}
    for col in df.columns:
        try:
            series = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
            if series.notnull().sum() > 0:
                datetime_analysis[col] = {
                    "min": str(series.min()),
                    "max": str(series.max()),
                    "missing_dates": int(series.isnull().sum())
                }
        except Exception:
            datetime_analysis[col] = {"error": "datetime parsing failed"}
    return datetime_analysis

def detect_ids_and_indexes(df: pd.DataFrame) -> dict:
    return {col: True for col in df.columns if df[col].is_unique}

def infer_target_column(df: pd.DataFrame) -> dict:
    possible_targets = [col for col in df.columns if 'target' in col.lower() or 'label' in col.lower()]
    target_analysis = {}
    for col in possible_targets:
        try:
            if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
                target_analysis[col] = {
                    "type": "regression",
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "mean": float(df[col].mean())
                }
            else:
                counts = df[col].dropna().astype(str).value_counts()
                target_analysis[col] = {
                    "type": "classification",
                    "class_counts": counts.to_dict()
                }
        except Exception:
            target_analysis[col] = {"error": "target analysis failed"}
    return target_analysis

def generate_column_recommendations(df: pd.DataFrame) -> dict:
    recommendations = {}
    for col in df.columns:
        try:
            recs = []
            if df[col].nunique(dropna=False) == 1:
                recs.append("drop - constant column")
            if df[col].isnull().mean() > 0.5:
                recs.append("high missing percentage")
            if pd.api.types.is_object_dtype(df[col]) and df[col].nunique() > 100:
                recs.append("high cardinality")
            if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
                s = df[col].dropna()
                if len(s) > 1 and s.skew() > 2:
                    recs.append("skewed distribution")
            if recs:
                recommendations[col] = recs
        except Exception:
            recommendations[col] = ["error generating recommendation"]
    return recommendations

def detect_feature_types(df: pd.DataFrame) -> dict:
    """
    Classify columns into feature types: continuous, discrete, categorical, text, boolean, datetime.
    """
    types = {}
    n = len(df)
    for col in df.columns:
        s = df[col]
        # boolean
        if pd.api.types.is_bool_dtype(s) or (
            pd.api.types.is_numeric_dtype(s)
            and set(s.dropna().unique()).issubset({0,1})
            and s.nunique(dropna=True) == 2
        ):
            types[col] = 'boolean'
        # datetime
        elif pd.api.types.is_datetime64_any_dtype(s):
            types[col] = 'datetime'
        else:
            parsed = pd.to_datetime(s, errors='coerce', infer_datetime_format=True)
            if parsed.notnull().sum() / n > 0.8:
                types[col] = 'datetime'
            elif pd.api.types.is_numeric_dtype(s):
                nn = s.dropna()
                if nn.empty:
                    types[col] = 'continuous'
                else:
                    uniq_ratio = nn.nunique() / n
                    is_int = np.allclose(nn % 1, 0)
                    types[col] = 'discrete' if is_int and uniq_ratio < 0.05 else 'continuous'
            else:
                nn = s.dropna().astype(str)
                uniq_ratio = nn.nunique() / n if n > 0 else 0
                avg_len = nn.str.len().mean() if not nn.empty else 0
                types[col] = 'text' if avg_len > 50 or uniq_ratio > 0.5 else 'categorical'
    return types

def analyze_continuous(df: pd.DataFrame, col: str) -> dict:
    series = pd.to_numeric(df[col], errors='coerce').dropna()
    return {
        'count': int(len(series)),
        'min': float(series.min()),
        'max': float(series.max()),
        'mean': float(series.mean()),
        'median': float(series.median()),
        'std': float(series.std()),
        'skew': float(skew(series)),
        'kurtosis': float(kurtosis(series)),
        'percentiles': {p: float(np.percentile(series, p)) for p in [5,25,50,75,95]},
        'coeff_variation': float(series.std()/series.mean()) if series.mean() != 0 else None,
        'zero_inflation': int((series == 0).sum())
    }

def analyze_discrete(df: pd.DataFrame, col: str) -> dict:
    series = pd.to_numeric(df[col], errors='coerce').dropna().astype(int)
    counts = series.value_counts()
    return {
        'count': int(len(series)),
        'unique_count': int(series.nunique()),
        'top_values': counts.head(5).to_dict(),
        'rare_values_ratio': float(counts[counts == 1].sum()/len(series)) if len(series) > 0 else None
    }

def analyze_categorical(df: pd.DataFrame, col: str) -> dict:
    series = df[col].dropna().astype(str)
    counts = series.value_counts()
    return {
        'count': int(len(series)),
        'unique_count': int(series.nunique()),
        'top_values': counts.head(5).to_dict(),
        'rare_categories_percentage': float(counts[counts == 1].sum()/len(series) * 100) if len(series) > 0 else None
    }

def analyze_text(df: pd.DataFrame, col: str) -> dict:
    series = df[col].dropna().astype(str)
    tokens = series.str.split().explode()
    return {
        'count': int(len(series)),
        'avg_length': float(series.str.len().mean()) if len(series) > 0 else None,
        'unique_tokens': int(tokens.nunique()) if not tokens.empty else 0,
        'top_tokens': tokens.value_counts().head(10).to_dict()
    }

def analyze_boolean(df: pd.DataFrame, col: str) -> dict:
    series = df[col]
    true_count = int((series == True).sum())
    false_count = int((series == False).sum())
    return {
        'count': int(series.notnull().sum()),
        'true_count': true_count,
        'false_count': false_count,
        'true_ratio': float(true_count/(true_count+false_count)) if (true_count+false_count) > 0 else None
    }

def analyze_datetime(df: pd.DataFrame, col: str) -> dict:
    series = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
    nn = series.dropna()
    stats = {
        'count': int(nn.count()),
        'min': str(nn.min()) if not nn.empty else None,
        'max': str(nn.max()) if not nn.empty else None,
        'missing_dates': int(series.isnull().sum())
    }
    if len(nn) > 1:
        stats['inferred_freq'] = pd.infer_freq(nn.sort_values())
    return stats
