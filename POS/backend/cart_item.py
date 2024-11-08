from typing import List, Optional
from pydantic import BaseModel

# Data model for detection results
class CartItem(BaseModel):
    item: str
    confidence: float
    top5_predictions: List[str]
    sandwich_item: Optional[str] = None
    sandwich_confidence: Optional[float] = None
    data_matrix_item: Optional[str] = None
    data_matrix: Optional[str] = None
    SKU: Optional[str] = None
    barcode: Optional[str] = None
    lookupcode: Optional[str] = None
    user_input: bool = False
    bbox: List[float]
    price: Optional[str] = None
