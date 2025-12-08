"""
Waste Bin API Routes
Endpoints for waste bin management
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from app.database import get_db
from app.schemas import WasteBinCreate, WasteBinUpdate, WasteBinResponse
from app.crud import (
    create_waste_bin, get_waste_bin, get_waste_bins,
    get_bins_by_category, update_waste_bin, delete_waste_bin
)
from app.models import WasteCategory

router = APIRouter(prefix="/bins", tags=["Waste Bins"])


@router.post("", response_model=WasteBinResponse, summary="Create waste bin")
def create_bin(
    bin_data: WasteBinCreate,
    db: Session = Depends(get_db)
):
    """Create a new waste bin"""
    return create_waste_bin(db, bin_data)


@router.get("", response_model=List[WasteBinResponse], summary="List waste bins")
def list_bins(
    skip: int = 0,
    limit: int = 100,
    active_only: bool = True,
    db: Session = Depends(get_db)
):
    """Get list of waste bins"""
    return get_waste_bins(db, skip, limit, active_only)


@router.get("/{bin_id}", response_model=WasteBinResponse, summary="Get waste bin")
def get_bin(
    bin_id: int,
    db: Session = Depends(get_db)
):
    """Get single waste bin by ID"""
    bin = get_waste_bin(db, bin_id)
    if not bin:
        raise HTTPException(status_code=404, detail="Waste bin not found")
    return bin


@router.get("/category/{category}", response_model=List[WasteBinResponse], summary="Get bins by category")
def get_bins_by_cat(
    category: WasteCategory,
    db: Session = Depends(get_db)
):
    """Get all bins for a specific waste category"""
    return get_bins_by_category(db, category)


@router.put("/{bin_id}", response_model=WasteBinResponse, summary="Update waste bin")
def update_bin(
    bin_id: int,
    bin_update: WasteBinUpdate,
    db: Session = Depends(get_db)
):
    """Update waste bin information"""
    bin = update_waste_bin(db, bin_id, bin_update)
    if not bin:
        raise HTTPException(status_code=404, detail="Waste bin not found")
    return bin


@router.delete("/{bin_id}", summary="Delete waste bin")
def delete_bin(
    bin_id: int,
    db: Session = Depends(get_db)
):
    """Delete (deactivate) waste bin"""
    success = delete_waste_bin(db, bin_id)
    if not success:
        raise HTTPException(status_code=404, detail="Waste bin not found")
    return {"message": "Waste bin deleted successfully"}
