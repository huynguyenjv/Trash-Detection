"""
Dashboard API Routes
Comprehensive statistics endpoints for the dashboard
"""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, cast, Date
from typing import List, Optional
from datetime import datetime, timedelta

from app.database import get_db
from app.models import Detection, DetectionSession, WasteBin, WasteCategory

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])


@router.get("/overview", summary="Get dashboard overview statistics")
def get_dashboard_overview(db: Session = Depends(get_db)):
    """
    Get comprehensive dashboard overview
    
    Returns:
    - Total counts by category
    - Today's statistics
    - Weekly statistics
    - Monthly statistics
    - Total sessions
    - Total bins
    """
    now = datetime.utcnow()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_start = now - timedelta(days=7)
    month_start = now - timedelta(days=30)
    
    # All-time totals from sessions
    all_sessions = db.query(DetectionSession).all()
    
    all_time = {
        'organic': sum(s.organic_count for s in all_sessions),
        'recyclable': sum(s.recyclable_count for s in all_sessions),
        'hazardous': sum(s.hazardous_count for s in all_sessions),
        'other': sum(s.other_count for s in all_sessions),
        'total': sum(s.total_detections for s in all_sessions),
        'sessions': len(all_sessions)
    }
    
    # Today's totals
    today_sessions = [s for s in all_sessions if s.started_at >= today_start]
    today = {
        'organic': sum(s.organic_count for s in today_sessions),
        'recyclable': sum(s.recyclable_count for s in today_sessions),
        'hazardous': sum(s.hazardous_count for s in today_sessions),
        'other': sum(s.other_count for s in today_sessions),
        'total': sum(s.total_detections for s in today_sessions),
        'sessions': len(today_sessions)
    }
    
    # Weekly totals
    week_sessions = [s for s in all_sessions if s.started_at >= week_start]
    weekly = {
        'organic': sum(s.organic_count for s in week_sessions),
        'recyclable': sum(s.recyclable_count for s in week_sessions),
        'hazardous': sum(s.hazardous_count for s in week_sessions),
        'other': sum(s.other_count for s in week_sessions),
        'total': sum(s.total_detections for s in week_sessions),
        'sessions': len(week_sessions)
    }
    
    # Monthly totals
    month_sessions = [s for s in all_sessions if s.started_at >= month_start]
    monthly = {
        'organic': sum(s.organic_count for s in month_sessions),
        'recyclable': sum(s.recyclable_count for s in month_sessions),
        'hazardous': sum(s.hazardous_count for s in month_sessions),
        'other': sum(s.other_count for s in month_sessions),
        'total': sum(s.total_detections for s in month_sessions),
        'sessions': len(month_sessions)
    }
    
    # Bin statistics
    bins = db.query(WasteBin).filter(WasteBin.is_active == 1).all()
    bins_by_category = {}
    for category in WasteCategory:
        bins_by_category[category.value] = len([b for b in bins if b.category == category])
    
    return {
        'all_time': all_time,
        'today': today,
        'weekly': weekly,
        'monthly': monthly,
        'bins': {
            'total': len(bins),
            'by_category': bins_by_category
        }
    }


@router.get("/by-category/{category}", summary="Get statistics by waste category")
def get_stats_by_category(
    category: str,
    days: int = Query(default=30, ge=1, le=365),
    db: Session = Depends(get_db)
):
    """
    Get detailed statistics for a specific waste category
    
    - **category**: organic, recyclable, hazardous, other
    - **days**: Number of days to look back (default: 30)
    """
    now = datetime.utcnow()
    start_date = now - timedelta(days=days)
    
    # Get detections for this category
    try:
        waste_category = WasteCategory(category)
    except ValueError:
        return {"error": f"Invalid category: {category}"}
    
    detections = db.query(Detection).filter(
        Detection.category == waste_category,
        Detection.detected_at >= start_date
    ).all()
    
    # Group by day
    daily_counts = {}
    for det in detections:
        day_key = det.detected_at.strftime('%Y-%m-%d')
        if day_key not in daily_counts:
            daily_counts[day_key] = 0
        daily_counts[day_key] += 1
    
    # Fill in missing days
    chart_data = []
    for i in range(days):
        day = (now - timedelta(days=days-1-i)).strftime('%Y-%m-%d')
        chart_data.append({
            'date': day,
            'count': daily_counts.get(day, 0)
        })
    
    # Get related bins
    related_bins = db.query(WasteBin).filter(
        WasteBin.category == waste_category,
        WasteBin.is_active == 1
    ).all()
    
    return {
        'category': category,
        'total_count': len(detections),
        'period_days': days,
        'daily_chart': chart_data,
        'avg_per_day': len(detections) / days if days > 0 else 0,
        'related_bins': [
            {
                'id': b.id,
                'name': b.name,
                'address': b.address,
                'latitude': b.latitude,
                'longitude': b.longitude,
                'current_fill': b.current_fill
            }
            for b in related_bins
        ]
    }


@router.get("/by-location", summary="Get statistics grouped by collection location")
def get_stats_by_location(
    days: int = Query(default=30, ge=1, le=365),
    db: Session = Depends(get_db)
):
    """
    Get waste statistics grouped by collection bin location
    """
    now = datetime.utcnow()
    start_date = now - timedelta(days=days)
    
    # Get all active bins
    bins = db.query(WasteBin).filter(WasteBin.is_active == 1).all()
    
    # Get all detections with location data
    detections = db.query(Detection).filter(
        Detection.detected_at >= start_date,
        Detection.latitude.isnot(None),
        Detection.longitude.isnot(None)
    ).all()
    
    # Calculate nearby detections for each bin (within ~100m radius)
    location_stats = []
    for bin in bins:
        # Simple distance calculation (approximate)
        nearby_detections = []
        for det in detections:
            lat_diff = abs(det.latitude - bin.latitude)
            lng_diff = abs(det.longitude - bin.longitude)
            # Roughly 0.001 degree ≈ 100m
            if lat_diff < 0.001 and lng_diff < 0.001:
                nearby_detections.append(det)
        
        # Count by category for this bin
        category_counts = {
            'organic': 0,
            'recyclable': 0,
            'hazardous': 0,
            'other': 0
        }
        for det in nearby_detections:
            category_counts[det.category.value] += 1
        
        location_stats.append({
            'bin_id': bin.id,
            'bin_name': bin.name,
            'address': bin.address,
            'category': bin.category.value,
            'latitude': bin.latitude,
            'longitude': bin.longitude,
            'current_fill': bin.current_fill,
            'total_nearby_detections': len(nearby_detections),
            'detections_by_category': category_counts
        })
    
    # Sort by total detections
    location_stats.sort(key=lambda x: x['total_nearby_detections'], reverse=True)
    
    return {
        'period_days': days,
        'total_bins': len(bins),
        'locations': location_stats
    }


@router.get("/daily-chart", summary="Get daily chart data for all categories")
def get_daily_chart(
    days: int = Query(default=30, ge=1, le=365),
    db: Session = Depends(get_db)
):
    """
    Get daily detection counts for chart visualization
    Returns data for all categories grouped by day
    """
    now = datetime.utcnow()
    start_date = now - timedelta(days=days)
    
    # Get all sessions in the period
    sessions = db.query(DetectionSession).filter(
        DetectionSession.started_at >= start_date
    ).all()
    
    # Group by day
    daily_data = {}
    for session in sessions:
        day_key = session.started_at.strftime('%Y-%m-%d')
        if day_key not in daily_data:
            daily_data[day_key] = {
                'organic': 0,
                'recyclable': 0,
                'hazardous': 0,
                'other': 0,
                'total': 0,
                'sessions': 0
            }
        daily_data[day_key]['organic'] += session.organic_count
        daily_data[day_key]['recyclable'] += session.recyclable_count
        daily_data[day_key]['hazardous'] += session.hazardous_count
        daily_data[day_key]['other'] += session.other_count
        daily_data[day_key]['total'] += session.total_detections
        daily_data[day_key]['sessions'] += 1
    
    # Build chart data with all days
    chart_data = []
    for i in range(days):
        day = (now - timedelta(days=days-1-i)).strftime('%Y-%m-%d')
        day_display = (now - timedelta(days=days-1-i)).strftime('%d/%m')
        
        data = daily_data.get(day, {
            'organic': 0,
            'recyclable': 0,
            'hazardous': 0,
            'other': 0,
            'total': 0,
            'sessions': 0
        })
        
        chart_data.append({
            'date': day,
            'display': day_display,
            **data
        })
    
    return {
        'period_days': days,
        'chart_data': chart_data
    }


@router.get("/hourly-chart", summary="Get hourly chart data for today")
def get_hourly_chart(db: Session = Depends(get_db)):
    """
    Get hourly detection counts for today
    """
    now = datetime.utcnow()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Get all detections today
    detections = db.query(Detection).filter(
        Detection.detected_at >= today_start
    ).all()
    
    # Group by hour
    hourly_data = {h: {'organic': 0, 'recyclable': 0, 'hazardous': 0, 'other': 0, 'total': 0} for h in range(24)}
    
    for det in detections:
        hour = det.detected_at.hour
        hourly_data[hour][det.category.value] += 1
        hourly_data[hour]['total'] += 1
    
    chart_data = []
    for hour in range(24):
        chart_data.append({
            'hour': f'{hour:02d}:00',
            **hourly_data[hour]
        })
    
    return {
        'date': today_start.strftime('%Y-%m-%d'),
        'chart_data': chart_data
    }


@router.get("/top-detections", summary="Get most frequent detection labels")
def get_top_detections(
    limit: int = Query(default=10, ge=1, le=50),
    days: int = Query(default=30, ge=1, le=365),
    db: Session = Depends(get_db)
):
    """
    Get the most frequently detected waste items
    """
    now = datetime.utcnow()
    start_date = now - timedelta(days=days)
    
    # Get all detections in the period
    detections = db.query(Detection).filter(
        Detection.detected_at >= start_date
    ).all()
    
    # Count by label
    label_counts = {}
    label_categories = {}
    for det in detections:
        if det.label not in label_counts:
            label_counts[det.label] = 0
            label_categories[det.label] = det.category.value
        label_counts[det.label] += 1
    
    # Sort and take top N
    sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:limit]
    
    return {
        'period_days': days,
        'top_items': [
            {
                'label': label,
                'count': count,
                'category': label_categories[label],
                'percentage': (count / len(detections) * 100) if detections else 0
            }
            for label, count in sorted_labels
        ]
    }


@router.get("/bin-statistics", summary="Get detailed bin statistics")
def get_bin_statistics(db: Session = Depends(get_db)):
    """
    Get detailed statistics for all waste bins
    """
    bins = db.query(WasteBin).filter(WasteBin.is_active == 1).all()
    
    # Calculate statistics per bin
    bin_stats = []
    for bin in bins:
        # Count nearby detections (all time)
        detections = db.query(Detection).filter(
            Detection.latitude.isnot(None),
            Detection.longitude.isnot(None)
        ).all()
        
        nearby_count = 0
        for det in detections:
            lat_diff = abs(det.latitude - bin.latitude) if det.latitude else 999
            lng_diff = abs(det.longitude - bin.longitude) if det.longitude else 999
            if lat_diff < 0.001 and lng_diff < 0.001:
                nearby_count += 1
        
        bin_stats.append({
            'id': bin.id,
            'name': bin.name,
            'category': bin.category.value,
            'address': bin.address,
            'latitude': bin.latitude,
            'longitude': bin.longitude,
            'capacity': bin.capacity,
            'current_fill': bin.current_fill,
            'fill_percentage': (bin.current_fill / bin.capacity * 100) if bin.capacity > 0 else 0,
            'last_emptied': bin.last_emptied.isoformat() if bin.last_emptied else None,
            'nearby_detections': nearby_count
        })
    
    # Group by category
    by_category = {}
    for cat in WasteCategory:
        cat_bins = [b for b in bin_stats if b['category'] == cat.value]
        by_category[cat.value] = {
            'count': len(cat_bins),
            'avg_fill': sum(b['fill_percentage'] for b in cat_bins) / len(cat_bins) if cat_bins else 0,
            'total_detections': sum(b['nearby_detections'] for b in cat_bins)
        }
    
    return {
        'total_bins': len(bins),
        'bins': bin_stats,
        'by_category': by_category
    }


@router.get("/sessions", summary="Get detection sessions history")
def get_sessions_history(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """
    Get detection sessions history with pagination
    """
    total = db.query(DetectionSession).count()
    sessions = db.query(DetectionSession).order_by(
        DetectionSession.started_at.desc()
    ).offset(skip).limit(limit).all()
    
    return {
        'total': total,
        'skip': skip,
        'limit': limit,
        'sessions': [
            {
                'id': s.id,
                'started_at': s.started_at.isoformat(),
                'ended_at': s.ended_at.isoformat() if s.ended_at else None,
                'duration_seconds': (s.ended_at - s.started_at).total_seconds() if s.ended_at else None,
                'total_detections': s.total_detections,
                'organic_count': s.organic_count,
                'recyclable_count': s.recyclable_count,
                'hazardous_count': s.hazardous_count,
                'other_count': s.other_count,
                'device_id': s.device_id
            }
            for s in sessions
        ]
    }


@router.get("/category-distribution", summary="Get waste category distribution")
def get_category_distribution(
    period: str = Query(default="all", regex="^(today|week|month|all)$"),
    db: Session = Depends(get_db)
):
    """
    Get waste category distribution for pie chart
    
    - **period**: today, week, month, all
    """
    now = datetime.utcnow()
    
    if period == "today":
        start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
    elif period == "week":
        start_date = now - timedelta(days=7)
    elif period == "month":
        start_date = now - timedelta(days=30)
    else:
        start_date = None
    
    # Get sessions
    query = db.query(DetectionSession)
    if start_date:
        query = query.filter(DetectionSession.started_at >= start_date)
    
    sessions = query.all()
    
    totals = {
        'organic': sum(s.organic_count for s in sessions),
        'recyclable': sum(s.recyclable_count for s in sessions),
        'hazardous': sum(s.hazardous_count for s in sessions),
        'other': sum(s.other_count for s in sessions)
    }
    
    total = sum(totals.values())
    
    distribution = []
    colors = {
        'organic': '#22c55e',
        'recyclable': '#3b82f6',
        'hazardous': '#ef4444',
        'other': '#6b7280'
    }
    labels = {
        'organic': 'Hữu cơ',
        'recyclable': 'Tái chế',
        'hazardous': 'Nguy hại',
        'other': 'Khác'
    }
    
    for cat, count in totals.items():
        distribution.append({
            'category': cat,
            'label': labels[cat],
            'count': count,
            'percentage': (count / total * 100) if total > 0 else 0,
            'color': colors[cat]
        })
    
    return {
        'period': period,
        'total': total,
        'distribution': distribution
    }


@router.get("/detections", summary="Get all detections with filters")
def get_all_detections(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=500),
    category: Optional[str] = Query(default=None, description="Filter by category: organic, recyclable, hazardous, other"),
    label: Optional[str] = Query(default=None, description="Filter by label (partial match)"),
    date_from: Optional[str] = Query(default=None, description="Filter from date (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(default=None, description="Filter to date (YYYY-MM-DD)"),
    session_id: Optional[int] = Query(default=None, description="Filter by session ID"),
    min_confidence: Optional[float] = Query(default=None, ge=0, le=1, description="Minimum confidence"),
    sort_by: str = Query(default="detected_at", description="Sort by: detected_at, label, confidence, category"),
    sort_order: str = Query(default="desc", description="Sort order: asc, desc"),
    db: Session = Depends(get_db)
):
    """
    Get all detections with comprehensive filtering and pagination
    
    Filters:
    - category: Filter by waste category
    - label: Filter by detection label (partial match)
    - date_from, date_to: Date range filter
    - session_id: Filter by detection session
    - min_confidence: Minimum confidence threshold
    
    Pagination:
    - skip: Number of records to skip
    - limit: Maximum records to return (max 500)
    
    Sorting:
    - sort_by: Field to sort by
    - sort_order: asc or desc
    """
    query = db.query(Detection)
    
    # Apply filters
    if category:
        try:
            waste_category = WasteCategory(category)
            query = query.filter(Detection.category == waste_category)
        except ValueError:
            pass
    
    if label:
        query = query.filter(Detection.label.ilike(f"%{label}%"))
    
    if date_from:
        try:
            from_date = datetime.strptime(date_from, "%Y-%m-%d")
            query = query.filter(Detection.detected_at >= from_date)
        except ValueError:
            pass
    
    if date_to:
        try:
            to_date = datetime.strptime(date_to, "%Y-%m-%d").replace(hour=23, minute=59, second=59)
            query = query.filter(Detection.detected_at <= to_date)
        except ValueError:
            pass
    
    if session_id:
        query = query.filter(Detection.session_id == session_id)
    
    if min_confidence is not None:
        query = query.filter(Detection.confidence >= min_confidence)
    
    # Get total count before pagination
    total = query.count()
    
    # Apply sorting
    sort_column = getattr(Detection, sort_by, Detection.detected_at)
    if sort_order == "asc":
        query = query.order_by(sort_column.asc())
    else:
        query = query.order_by(sort_column.desc())
    
    # Apply pagination
    detections = query.offset(skip).limit(limit).all()
    
    # Get unique labels for filter dropdown
    all_labels = db.query(Detection.label).distinct().all()
    unique_labels = sorted(set(label[0] for label in all_labels))
    
    return {
        'total': total,
        'skip': skip,
        'limit': limit,
        'has_more': skip + limit < total,
        'filters': {
            'category': category,
            'label': label,
            'date_from': date_from,
            'date_to': date_to,
            'session_id': session_id,
            'min_confidence': min_confidence
        },
        'available_labels': unique_labels,
        'detections': [
            {
                'id': d.id,
                'session_id': d.session_id,
                'label': d.label,
                'category': d.category.value,
                'confidence': d.confidence,
                'bbox': [d.bbox_x, d.bbox_y, d.bbox_width, d.bbox_height],
                'latitude': d.latitude,
                'longitude': d.longitude,
                'detected_at': d.detected_at.isoformat(),
                'tracking_data': d.tracking_data
            }
            for d in detections
        ]
    }


@router.get("/detection-labels", summary="Get all unique detection labels")
def get_detection_labels(db: Session = Depends(get_db)):
    """
    Get all unique detection labels with counts
    """
    detections = db.query(Detection).all()
    
    label_counts = {}
    label_categories = {}
    
    for det in detections:
        if det.label not in label_counts:
            label_counts[det.label] = 0
            label_categories[det.label] = det.category.value
        label_counts[det.label] += 1
    
    sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
    
    return {
        'total_labels': len(sorted_labels),
        'labels': [
            {
                'label': label,
                'count': count,
                'category': label_categories[label]
            }
            for label, count in sorted_labels
        ]
    }
