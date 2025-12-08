"""
Statistics API Routes
Endpoints for waste statistics
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime, timedelta

from app.database import get_db
from app.schemas import WasteStatsResponse, StatsResponse
from app.crud import get_stats_by_period, get_recent_sessions
from app.services import WasteManager

router = APIRouter(prefix="/stats", tags=["Statistics"])

# Initialize waste manager
waste_manager = WasteManager()


@router.get("/current", response_model=StatsResponse, summary="Get current statistics")
def get_current_stats():
    """
    Get current session statistics
    
    Returns totals and recent detections from in-memory manager
    """
    return waste_manager.get_stats()


@router.get("/history/{period_type}", response_model=List[WasteStatsResponse], summary="Get historical statistics")
def get_historical_stats(
    period_type: str,
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """
    Get historical statistics by period
    
    - **period_type**: hourly, daily, weekly, monthly
    - **limit**: Number of periods to return (default: 10)
    """
    return get_stats_by_period(db, period_type, limit)


@router.get("/summary", summary="Get statistics summary")
def get_stats_summary(db: Session = Depends(get_db)):
    """
    Get comprehensive statistics summary
    
    Returns:
    - Current session stats
    - Today's totals
    - Week's totals
    - All-time totals
    """
    # Current session (in-memory)
    current = waste_manager.get_stats()
    
    # Recent sessions from DB
    recent_sessions = get_recent_sessions(db, limit=100)
    
    # Calculate today's totals
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    today_sessions = [s for s in recent_sessions if s.started_at >= today_start]
    today_totals = {
        'organic': sum(s.organic_count for s in today_sessions),
        'recyclable': sum(s.recyclable_count for s in today_sessions),
        'hazardous': sum(s.hazardous_count for s in today_sessions),
        'other': sum(s.other_count for s in today_sessions),
        'total': sum(s.total_detections for s in today_sessions)
    }
    
    # Calculate week's totals
    week_start = datetime.now() - timedelta(days=7)
    week_sessions = [s for s in recent_sessions if s.started_at >= week_start]
    week_totals = {
        'organic': sum(s.organic_count for s in week_sessions),
        'recyclable': sum(s.recyclable_count for s in week_sessions),
        'hazardous': sum(s.hazardous_count for s in week_sessions),
        'other': sum(s.other_count for s in week_sessions),
        'total': sum(s.total_detections for s in week_sessions)
    }
    
    # All-time totals
    all_time_totals = {
        'organic': sum(s.organic_count for s in recent_sessions),
        'recyclable': sum(s.recyclable_count for s in recent_sessions),
        'hazardous': sum(s.hazardous_count for s in recent_sessions),
        'other': sum(s.other_count for s in recent_sessions),
        'total': sum(s.total_detections for s in recent_sessions),
        'sessions': len(recent_sessions)
    }
    
    return {
        'current': current['totals'],
        'today': today_totals,
        'week': week_totals,
        'all_time': all_time_totals
    }


@router.post("/reset", summary="Reset current statistics")
def reset_stats():
    """Reset in-memory statistics counter"""
    waste_manager.reset()
    return {"message": "Statistics reset successfully"}
