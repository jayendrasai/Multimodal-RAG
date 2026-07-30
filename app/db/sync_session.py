from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.config import get_settings

settings = get_settings()

# Convert the async Postgres URL to a sync URL for psycopg2
sync_db_url = str(settings.DATABASE_URL).replace("+asyncpg", "")

engine = create_engine(
    sync_db_url,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_timeout=settings.DB_POOL_TIMEOUT,
    pool_pre_ping=True, 
)

SyncSessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)