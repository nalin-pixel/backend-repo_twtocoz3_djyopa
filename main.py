import os
from typing import Optional, List, Literal, Dict, Any
from datetime import datetime
from hashlib import sha256

from fastapi import FastAPI, Depends, HTTPException, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from bson import ObjectId

from database import db, create_document, get_documents

# Firebase Admin for token verification
import firebase_admin
from firebase_admin import auth as fb_auth, credentials

# Initialize Firebase Admin SDK once if not already
if not firebase_admin._apps:
    cred_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
    if cred_json:
        try:
            cred = credentials.Certificate(eval(cred_json))  # expects dict-like JSON in env
            firebase_admin.initialize_app(cred)
        except Exception:
            firebase_admin.initialize_app()  # fallback to default credentials
    else:
        try:
            firebase_admin.initialize_app()
        except Exception:
            pass

app = FastAPI(title="Rental Agreement Management API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------
# Helpers
# ------------------------
class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if isinstance(v, ObjectId):
            return v
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

Role = Literal["landlord", "tenant"]

class UserModel(BaseModel):
    auth_uid: str
    email: str
    display_name: Optional[str] = None
    role: Role = "tenant"
    phone: Optional[str] = None

class ListingModel(BaseModel):
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

class ListingOut(ListingModel):
    id: str
    landlord_uid: str

class RentalRequestModel(BaseModel):
    listing_id: str
    message: Optional[str] = None

class RentalRequestOut(BaseModel):
    id: str
    listing_id: str
    tenant_uid: str
    status: Literal["pending", "approved", "rejected"]
    message: Optional[str] = None
    created_at: Optional[datetime] = None

class AgreementModel(BaseModel):
    id: str
    listing_id: str
    landlord_uid: str
    tenant_uid: str
    start_date: datetime
    end_date: datetime
    monthly_rent: float
    terms: Optional[str] = None
    hash: Optional[str] = None


def verify_token(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    try:
        scheme, token = authorization.split(" ")
        if scheme.lower() != "bearer":
            raise ValueError("Invalid auth scheme")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid Authorization header format")

    try:
        decoded = fb_auth.verify_id_token(token)
        return decoded  # contains uid, email, etc.
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)[:100]}")


def get_current_user(decoded: Dict[str, Any] = Depends(verify_token)) -> Dict[str, Any]:
    uid = decoded.get("uid")
    user_doc = db["user"].find_one({"auth_uid": uid})
    role = None
    if user_doc:
        role = user_doc.get("role")
    return {"uid": uid, "email": decoded.get("email"), "role": role}


@app.get("/")
def root():
    return {"name": "Rental Agreement Management API", "status": "ok"}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set",
        "database_name": "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set",
        "connection_status": "Not Connected",
        "collections": []
    }
    try:
        if db is not None:
            response["database"] = "✅ Connected"
            response["connection_status"] = "Connected"
            response["collections"] = db.list_collection_names()
    except Exception as e:
        response["database"] = f"⚠️ {str(e)[:80]}"
    return response

# ------------------------
# Users
# ------------------------
@app.get("/users/me")
def get_me(current=Depends(get_current_user)):
    uid = current["uid"]
    doc = db["user"].find_one({"auth_uid": uid})
    if not doc:
        return {"exists": False, "user": None}
    doc["id"] = str(doc.pop("_id"))
    return {"exists": True, "user": doc}


@app.post("/users/me")
def upsert_me(payload: UserModel, current=Depends(get_current_user)):
    if payload.auth_uid != current["uid"]:
        raise HTTPException(status_code=403, detail="UID mismatch")
    existing = db["user"].find_one({"auth_uid": payload.auth_uid})
    data = payload.model_dump()
    data["updated_at"] = datetime.utcnow()
    if existing:
        db["user"].update_one({"_id": existing["_id"]}, {"$set": data})
        doc = db["user"].find_one({"_id": existing["_id"]})
    else:
        data["created_at"] = datetime.utcnow()
        inserted_id = db["user"].insert_one(data).inserted_id
        doc = db["user"].find_one({"_id": inserted_id})
    doc["id"] = str(doc.pop("_id"))
    return doc

# ------------------------
# Listings
# ------------------------
@app.post("/listings", response_model=ListingOut)
def create_listing(payload: ListingModel, current=Depends(get_current_user)):
    if current.get("role") != "landlord":
        raise HTTPException(status_code=403, detail="Only landlords can create listings")
    data = payload.model_dump()
    data["landlord_uid"] = current["uid"]
    data["created_at"] = datetime.utcnow()
    data["updated_at"] = datetime.utcnow()
    inserted_id = db["listing"].insert_one(data).inserted_id
    return ListingOut(id=str(inserted_id), landlord_uid=current["uid"], **payload.model_dump())


@app.get("/listings")
def list_listings(
    q: Optional[str] = Query(None, description="Search query across title, description, address, city, state"),
    city: Optional[str] = None,
    state: Optional[str] = None,
    min_rent: Optional[float] = None,
    max_rent: Optional[float] = None,
    bedrooms: Optional[int] = None,
    sort_by: str = Query("created_at", regex="^(created_at|rent|bedrooms)$"),
    sort_dir: str = Query("desc", regex="^(asc|desc)$"),
    page: int = 1,
    page_size: int = 10,
):
    filter_: Dict[str, Any] = {"is_active": True}
    if city:
        filter_["city"] = {"$regex": f"^{city}$", "$options": "i"}
    if state:
        filter_["state"] = {"$regex": f"^{state}$", "$options": "i"}
    if min_rent is not None or max_rent is not None:
        price_cond: Dict[str, Any] = {}
        if min_rent is not None:
            price_cond["$gte"] = min_rent
        if max_rent is not None:
            price_cond["$lte"] = max_rent
        filter_["rent"] = price_cond
    if bedrooms is not None:
        filter_["bedrooms"] = {"$gte": bedrooms}
    if q:
        filter_["$or"] = [
            {"title": {"$regex": q, "$options": "i"}},
            {"description": {"$regex": q, "$options": "i"}},
            {"address": {"$regex": q, "$options": "i"}},
            {"city": {"$regex": q, "$options": "i"}},
            {"state": {"$regex": q, "$options": "i"}},
        ]
    total = db["listing"].count_documents(filter_)
    sort_dir_num = -1 if sort_dir == "desc" else 1
    cursor = db["listing"].find(filter_).sort(sort_by, sort_dir_num).skip((page - 1) * page_size).limit(page_size)
    items = []
    for doc in cursor:
        doc_out = {**doc}
        doc_out["id"] = str(doc_out.pop("_id"))
        items.append(doc_out)
    return {"items": items, "total": total, "page": page, "page_size": page_size}


@app.get("/listings/{listing_id}")
def get_listing(listing_id: str):
    doc = db["listing"].find_one({"_id": PyObjectId.validate(listing_id)})
    if not doc:
        raise HTTPException(status_code=404, detail="Listing not found")
    doc["id"] = str(doc.pop("_id"))
    return doc


@app.put("/listings/{listing_id}")
def update_listing(listing_id: str, payload: ListingModel, current=Depends(get_current_user)):
    listing = db["listing"].find_one({"_id": PyObjectId.validate(listing_id)})
    if not listing:
        raise HTTPException(status_code=404, detail="Listing not found")
    if listing.get("landlord_uid") != current["uid"]:
        raise HTTPException(status_code=403, detail="Only the owner can update this listing")
    data = payload.model_dump()
    data["updated_at"] = datetime.utcnow()
    db["listing"].update_one({"_id": listing["_id"]}, {"$set": data})
    updated = db["listing"].find_one({"_id": listing["_id"]})
    updated["id"] = str(updated.pop("_id"))
    return updated


@app.delete("/listings/{listing_id}")
def delete_listing(listing_id: str, current=Depends(get_current_user)):
    listing = db["listing"].find_one({"_id": PyObjectId.validate(listing_id)})
    if not listing:
        raise HTTPException(status_code=404, detail="Listing not found")
    if listing.get("landlord_uid") != current["uid"]:
        raise HTTPException(status_code=403, detail="Only the owner can delete this listing")
    db["listing"].delete_one({"_id": listing["_id"]})
    return {"deleted": True}

# ------------------------
# Rental Requests
# ------------------------
@app.post("/requests", response_model=RentalRequestOut)
def create_request(payload: RentalRequestModel, current=Depends(get_current_user)):
    if current.get("role") != "tenant":
        raise HTTPException(status_code=403, detail="Only tenants can create rental requests")
    # Ensure listing exists and is active
    listing = db["listing"].find_one({"_id": PyObjectId.validate(payload.listing_id)})
    if not listing or not listing.get("is_active", True):
        raise HTTPException(status_code=404, detail="Listing not available")
    doc = {
        "listing_id": payload.listing_id,
        "tenant_uid": current["uid"],
        "status": "pending",
        "message": payload.message,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }
    rid = db["rentalrequest"].insert_one(doc).inserted_id
    return RentalRequestOut(id=str(rid), listing_id=payload.listing_id, tenant_uid=current["uid"], status="pending", message=payload.message, created_at=doc["created_at"]) 


@app.get("/requests")
def get_requests(current=Depends(get_current_user)):
    role = current.get("role")
    uid = current["uid"]
    if role == "tenant":
        cursor = db["rentalrequest"].find({"tenant_uid": uid}).sort("created_at", -1)
    elif role == "landlord":
        # find listings owned by landlord
        listing_ids = [str(x["_id"]) for x in db["listing"].find({"landlord_uid": uid}, {"_id": 1})]
        cursor = db["rentalrequest"].find({"listing_id": {"$in": listing_ids}}).sort("created_at", -1)
    else:
        raise HTTPException(status_code=403, detail="Role not set")
    items = []
    for doc in cursor:
        doc["id"] = str(doc.pop("_id"))
        items.append(doc)
    return {"items": items}


class RequestDecision(BaseModel):
    decision: Literal["approve", "reject"]
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    monthly_rent: Optional[float] = None
    terms: Optional[str] = None


@app.post("/requests/{request_id}/decision")
def decide_request(request_id: str, payload: RequestDecision, current=Depends(get_current_user)):
    req = db["rentalrequest"].find_one({"_id": PyObjectId.validate(request_id)})
    if not req:
        raise HTTPException(status_code=404, detail="Request not found")
    listing = db["listing"].find_one({"_id": PyObjectId.validate(req["listing_id"])})
    if not listing:
        raise HTTPException(status_code=404, detail="Listing not found")
    if listing.get("landlord_uid") != current["uid"]:
        raise HTTPException(status_code=403, detail="Only the listing owner can decide")

    status = "approved" if payload.decision == "approve" else "rejected"
    db["rentalrequest"].update_one({"_id": req["_id"]}, {"$set": {"status": status, "updated_at": datetime.utcnow()}})

    result: Dict[str, Any] = {"status": status}

    if status == "approved":
        for field in [payload.start_date, payload.end_date, payload.monthly_rent]:
            if field is None:
                raise HTTPException(status_code=400, detail="start_date, end_date, monthly_rent required on approval")
        agreement = {
            "listing_id": req["listing_id"],
            "landlord_uid": listing["landlord_uid"],
            "tenant_uid": req["tenant_uid"],
            "start_date": payload.start_date,
            "end_date": payload.end_date,
            "monthly_rent": payload.monthly_rent,
            "terms": payload.terms,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }
        # simple deterministic hash for verification placeholder
        hash_input = f"{agreement['listing_id']}|{agreement['landlord_uid']}|{agreement['tenant_uid']}|{agreement['start_date'].isoformat()}|{agreement['end_date'].isoformat()}|{agreement['monthly_rent']}"
        agreement["hash"] = sha256(hash_input.encode()).hexdigest()
        ag_id = db["agreement"].insert_one(agreement).inserted_id
        result["agreement_id"] = str(ag_id)
        result["hash"] = agreement["hash"]

    return result

# ------------------------
# Agreements
# ------------------------
@app.get("/agreements")
def my_agreements(current=Depends(get_current_user)):
    uid = current["uid"]
    role = current.get("role")
    filt = {"tenant_uid": uid} if role == "tenant" else {"landlord_uid": uid}
    items = []
    for doc in db["agreement"].find(filt).sort("created_at", -1):
        doc["id"] = str(doc.pop("_id"))
        items.append(doc)
    return {"items": items}


@app.get("/agreements/{agreement_id}")
def get_agreement(agreement_id: str, current=Depends(get_current_user)):
    doc = db["agreement"].find_one({"_id": PyObjectId.validate(agreement_id)})
    if not doc:
        raise HTTPException(status_code=404, detail="Agreement not found")
    if doc.get("tenant_uid") != current["uid"] and doc.get("landlord_uid") != current["uid"]:
        raise HTTPException(status_code=403, detail="Not authorized to view this agreement")
    doc["id"] = str(doc.pop("_id"))
    return doc


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
