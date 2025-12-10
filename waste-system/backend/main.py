"""
Main Application Entry Point
FastAPI application with database initialization
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from app.config import get_settings
from app.database import init_db, SessionLocal
from app.api import detection, bins, stats, websocket, routing
from app.models import WasteBin, WasteCategory

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
    """Add sample waste bins if none exist"""
    db = SessionLocal()
    try:
        # Check if bins already exist
        existing_bins = db.query(WasteBin).count()
        if existing_bins > 0:
            logger.info(f"📍 Found {existing_bins} existing bins, skipping seed")
            return
        
        # Sample bins for Ho Chi Minh City
        sample_bins = [
            WasteBin(
                name="Central Waste Bin - District 1",
                category=WasteCategory.OTHER,
                latitude=10.8231,
                longitude=106.6297,
                address="123 Nguyen Hue, District 1, HCMC",
                capacity=100,
                current_fill=25,
                is_active=True
            ),
            WasteBin(
                name="Recycling Center - District 3",
                category=WasteCategory.RECYCLABLE,
                latitude=10.7831,
                longitude=106.6797,
                address="456 Le Van Sy, District 3, HCMC",
                capacity=200,
                current_fill=45,
                is_active=True
            ),
            WasteBin(
                name="Organic Waste Bin - Binh Thanh",
                category=WasteCategory.ORGANIC,
                latitude=10.8031,
                longitude=106.7097,
                address="789 Xo Viet Nghe Tinh, Binh Thanh, HCMC",
                capacity=150,
                current_fill=60,
                is_active=True
            ),
            WasteBin(
                name="Hazardous Waste Facility - District 7",
                category=WasteCategory.HAZARDOUS,
                latitude=10.7331,
                longitude=106.7197,
                address="101 Nguyen Van Linh, District 7, HCMC",
                capacity=50,
                current_fill=10,
                is_active=True
            ),
            WasteBin(
                name="General Bin - Thu Duc",
                category=WasteCategory.OTHER,
                latitude=10.8531,
                longitude=106.7597,
                address="202 Vo Van Ngan, Thu Duc, HCMC",
                capacity=120,
                current_fill=35,
                is_active=True
            ),
        ]
        
        for bin_data in sample_bins:
            db.add(bin_data)
        
        db.commit()
        logger.info(f"✅ Seeded {len(sample_bins)} sample waste bins")
        
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level.lower()
    )
