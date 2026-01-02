"""
Main Application Entry Point
FastAPI application with database initialization
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from app.config import get_settings
from app.database import init_db, SessionLocal
from app.api import detection, bins, stats, websocket, routing, dashboard, locations
from app.models import WasteBin, WasteCategory
from app.data import HCM_WASTE_LOCATIONS

# Load settings
settings = get_settings()

# Configure logging
logging.basicConfig(
    level=settings.log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Smart Waste Detection System",
    description="AI-powered waste detection and management system",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    logger.info("🚀 Starting Smart Waste Detection System...")
    logger.info(f"📊 Database: {settings.database_url}")
    
    # Initialize database (create tables)
    try:
        init_db()
        logger.info("✅ Database initialized successfully")
        
        # Seed sample bins if database is empty
        await seed_sample_bins()
        
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        raise
    
    logger.info("✅ Application started successfully")


async def seed_sample_bins():
    """Add real HCM waste collection locations if none exist"""
    db = SessionLocal()
    try:
        # Check if bins already exist
        existing_bins = db.query(WasteBin).count()
        if existing_bins > 0:
            logger.info(f"📍 Found {existing_bins} existing bins, skipping seed")
            return
        
        # Map waste types from string to WasteCategory enum
        category_map = {
            "organic": WasteCategory.ORGANIC,
            "recyclable": WasteCategory.RECYCLABLE,
            "hazardous": WasteCategory.HAZARDOUS,
            "other": WasteCategory.OTHER
        }
        
        # Create bins from real HCM waste locations data
        bins_created = 0
        for location in HCM_WASTE_LOCATIONS:
            # Determine primary category based on waste_types
            waste_types = location.get("waste_types", ["other"])
            
            # Priority: hazardous > recyclable > organic > other
            if "hazardous" in waste_types:
                primary_category = WasteCategory.HAZARDOUS
            elif "recyclable" in waste_types:
                primary_category = WasteCategory.RECYCLABLE
            elif "organic" in waste_types:
                primary_category = WasteCategory.ORGANIC
            else:
                primary_category = WasteCategory.OTHER
            
            # Calculate initial fill level (random for demo)
            import random
            capacity = location.get("capacity_tons_per_day", 100)
            current_fill = random.randint(10, 80)
            
            bin_data = WasteBin(
                name=location["name"],
                category=primary_category,
                latitude=location["latitude"],
                longitude=location["longitude"],
                address=location["address"],
                capacity=capacity,
                current_fill=current_fill,
                is_active=location.get("status", "active") == "active"
            )
            db.add(bin_data)
            bins_created += 1
        
        db.commit()
        logger.info(f"✅ Seeded {bins_created} real HCM waste collection locations")
        
    except Exception as e:
        logger.error(f"❌ Error seeding bins: {e}")
        db.rollback()
    finally:
        db.close()


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("👋 Shutting down Smart Waste Detection System...")


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint - API information"""
    return {
        "name": "Smart Waste Detection System API",
        "version": "2.0.0",
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "detection": "/detection",
            "bins": "/bins",
            "stats": "/stats",
            "websocket_detect": "/ws/detect",
            "websocket_stats": "/ws/stats"
        }
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "database": "connected"
    }


# Include routers
app.include_router(detection.router)
app.include_router(bins.router)
app.include_router(stats.router)
app.include_router(websocket.router)
app.include_router(routing.router)
app.include_router(dashboard.router)
app.include_router(locations.router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level.lower()
    )
