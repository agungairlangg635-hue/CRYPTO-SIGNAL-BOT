"""Database connection setup."""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from pathlib import Path
import os
from dotenv import load_dotenv

from .models import Base

load_dotenv()

DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///./data/crypto_bot.db')

# Pastikan folder data ada
if 'sqlite' in DATABASE_URL:
    db_path = DATABASE_URL.replace('sqlite:///', '')
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True)
SessionLocal = scoped_session(sessionmaker(bind=engine))


def init_db():
    """Initialize database - buat semua tables."""
    print("  Initializing database...")
    Base.metadata.create_all(bind=engine)
    print(f" Database ready at: {DATABASE_URL}")


def get_session():
    """Get database session."""
    return SessionLocal()


if __name__ == "__main__":
    init_db()