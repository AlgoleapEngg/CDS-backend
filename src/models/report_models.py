from typing import List, Dict, Union
from pydantic import BaseModel
from .narrative_models import Narrative

class DatasetOverview(BaseModel):
    num_rows: int
    num_columns: int
    column_names: List[str]
    data_types: Dict[str, str]
    memory_usage_mb: float

class DataQuality(BaseModel):
    missing_values: Dict[str, int]
    missing_value_percentage: Dict[str, float]
    duplicate_rows: int
    constant_columns: List[str]
    mixed_type_columns: List[str]

class NumericStats(BaseModel):
    min: float
    max: float
    mean: float
    median: float
    std: float
    skew: float
    kurtosis: float

class CategoricalStats(BaseModel):
    unique_count: int
    top_values: Dict[str, int]

ValueDistribution = Union[NumericStats, CategoricalStats]

class AnalysisDetails(BaseModel):
    dataset_overview: DatasetOverview
    data_quality: DataQuality
    value_distributions: Dict[str, ValueDistribution]
    cardinality: Dict[str, int]
    outliers: Dict[str, int]
    correlations: Dict[str, Dict[str, float]]
    datetime_analysis: Dict[str, Dict[str, Union[str, int, None]]]
    id_detection: Dict[str, bool]
    target_inference: Dict[str, Dict[str, Union[str, float, Dict[str,int]]]]
    column_recommendations: Dict[str, List[str]]
    feature_based_analysis: Dict[str, Union[Dict[str,str], Dict[str,dict]]]

class Report(BaseModel):
    success: bool
    errors: List[str]
    analysis: AnalysisDetails
    narrative: Narrative
