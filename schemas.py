"""
Database Schemas for Rental App

Each Pydantic model becomes a MongoDB collection using the lowercase of the class
name. For example, class Listing -> "listing" collection.
"""

from typing import Optional, List, Literal
from pydantic import BaseModel, Field
from datetime import datetime

Role = Literal["landlord", "tenant"]

class User(BaseModel):
    auth_uid: str = Field(..., description="Firebase Auth UID")
    email: str
    display_name: Optional[str] = None
    role: Role = Field("tenant", description="User role: landlord or tenant")
    phone: Optional[str] = None

class Listing(BaseModel):
    landlord_uid: str = Field(..., description="Owner (Firebase UID)")
    title: str
    description: Optional[str] = None
    address: str
    city: str
    state: str
    rent: float
    bedrooms: int = 1
    bathrooms: float = 1
    amenities: List[str] = []
    available_from: Optional[datetime] = None
    is_active: bool = True

class RentalRequest(BaseModel):
    listing_id: str
    tenant_uid: str
    status: Literal["pending", "approved", "rejected"] = "pending"
    message: Optional[str] = None

class Agreement(BaseModel):
    listing_id: str
    landlord_uid: str
    tenant_uid: str
    start_date: datetime
    end_date: datetime
    monthly_rent: float
    terms: Optional[str] = None
    hash: Optional[str] = None  # placeholder for future blockchain verification
