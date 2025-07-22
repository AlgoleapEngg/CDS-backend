from pydantic import BaseModel

class Narrative(BaseModel):
    dataset_snapshot: str
    data_quality: str
    distributions: str
    correlations: str
    recommendations: str
    customer_highlights: str
