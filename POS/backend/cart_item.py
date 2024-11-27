from typing import List, Optional
from pydantic import BaseModel, ConfigDict

# Data model for detection results
class CartItem(BaseModel):
    model_config = ConfigDict(frozen=False)

    display_name: str = 'Unknown'
    item: str
    confidence: float
    top5_predictions: List[str]
    sandwich_item: Optional[str] = None
    sandwich_confidence: Optional[float] = None
    data_matrix: Optional[str] = None
    SKU: Optional[int] = None
    barcode: Optional[int] = None
    lookupcode: Optional[int] = None
    user_input: bool = False
    bbox: List[float]
    price: float = 0.00