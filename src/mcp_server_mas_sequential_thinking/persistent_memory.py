"""Persistent memory management with SQLAlchemy and memory pruning."""

import os
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    create_engine,
    desc,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker
from sqlalchemy.pool import StaticPool

from .constants import DatabaseConstants, DefaultSettings
from .logging_config import get_logger
from .models import ThoughtData

logger = get_logger(__name__)


class Base(DeclarativeBase):
    """Base class for SQLAlchemy models with modern typing support."""



class SessionRecord(Base):
    """Database model for session storage."""

    __tablename__ = "sessions"

    id = Column(String, primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    total_thoughts = Column(Integer, default=0)
    total_cost = Column(Float, default=0.0)
    provider = Column(String, default="deepseek")

    # Relationship to thoughts
    thoughts = relationship(
        "ThoughtRecord", back_populates="session", cascade="all, delete-orphan"
    )

    # Index for performance
    __table_args__ = (
        Index("ix_session_created", "created_at"),
        Index("ix_session_updated", "updated_at"),
    )


class ThoughtRecord(Base):
    """Database model for thought storage."""

    __tablename__ = "thoughts"

    id = Column(Integer, primary_key=True)
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False)
    thought_number = Column(Integer, nullable=False)
    thought = Column(Text, nullable=False)
    total_thoughts = Column(Integer, nullable=False)
    next_needed = Column(Boolean, nullable=False)
    branch_from = Column(Integer, nullable=True)
    branch_id = Column(String, nullable=True)

    # Processing metadata
    processing_strategy = Column(String, nullable=True)
    complexity_score = Column(Float, nullable=True)
    estimated_cost = Column(Float, nullable=True)
    actual_cost = Column(Float, nullable=True)
    token_usage = Column(Integer, nullable=True)
    processing_time = Column(Float, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)

    # Response data
    response = Column(Text, nullable=True)
    specialist_used = Column(JSON, nullable=True)  # List of specialists used

    # Relationship
    session = relationship("SessionRecord", back_populates="thoughts")

    # Indexes for performance
    __table_args__ = (
        Index("ix_thought_session", "session_id"),
        Index("ix_thought_number", "thought_number"),
        Index("ix_thought_created", "created_at"),
        Index("ix_thought_branch", "branch_from", "branch_id"),
    )

    def to_thought_data(self) -> ThoughtData:
        """Convert database record to ThoughtData model."""
        return ThoughtData(
            thought=str(self.thought),
            thought_number=int(self.thoughtNumber),
            total_thoughts=int(self.totalThoughts),
            next_needed=bool(self.nextThoughtNeeded),
            branch_from=int(self.branchFromThought) if self.branchFromThought is not None else None,
            branch_id=str(self.branchId) if self.branchId is not None else None,
            is_revision=False,
            branchFromThought=None,
            needs_more=True,
        )


class BranchRecord(Base):
    """Database model for branch tracking."""

    __tablename__ = "branches"

    id = Column(Integer, primary_key=True)
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False)
    branch_id = Column(String, nullable=False)
    parent_thought = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    thought_count = Column(Integer, default=0)

    __table_args__ = (
        Index("ix_branch_session", "session_id"),
        Index("ix_branch_id", "branch_id"),
    )


class UsageMetrics(Base):
    """Database model for usage tracking and cost optimization."""

    __tablename__ = "usage_metrics"

    id = Column(Integer, primary_key=True)
    date = Column(DateTime, default=datetime.utcnow)
    provider = Column(String, nullable=False)
    processing_strategy = Column(String, nullable=False)
    complexity_level = Column(String, nullable=False)

    # Metrics
    thought_count = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    total_cost = Column(Float, default=0.0)
    avg_processing_time = Column(Float, default=0.0)
    success_rate = Column(Float, default=1.0)

    # Performance tracking
    token_efficiency = Column(Float, default=0.0)  # quality/token ratio
    cost_effectiveness = Column(Float, default=0.0)  # quality/cost ratio

    __table_args__ = (
        Index("ix_metrics_date", "date"),
        Index("ix_metrics_provider", "provider"),
        Index("ix_metrics_strategy", "processing_strategy"),
    )


class PersistentMemoryManager:
    """Manages persistent storage and memory pruning."""

    def __init__(self, database_url: str | None = None):
        """Initialize persistent memory manager."""
        # Default to local SQLite database
        if database_url is None:
            db_dir = Path.home() / ".sequential_thinking" / "data"
            db_dir.mkdir(parents=True, exist_ok=True)
            database_url = f"sqlite:///{db_dir}/memory.db"

        # Configure engine with connection pooling
        if database_url.startswith("sqlite"):
            # SQLite-specific configuration
            self.engine = create_engine(
                database_url,
                poolclass=StaticPool,
                connect_args={"check_same_thread": False},
                echo=False,
            )
        else:
            # PostgreSQL/other database configuration
            self.engine = create_engine(
                database_url,
                pool_pre_ping=True,
                pool_size=DatabaseConstants.CONNECTION_POOL_SIZE,
                max_overflow=DatabaseConstants.CONNECTION_POOL_OVERFLOW,
            )

        # Create tables
        Base.metadata.create_all(self.engine)

        # Session factory
        self.SessionLocal = sessionmaker(bind=self.engine)

        logger.info(f"Persistent memory initialized with database: {database_url}")

    def create_session(
        self, session_id: str, provider: str = DefaultSettings.DEFAULT_PROVIDER
    ) -> None:
        """Create a new session record."""
        with self.SessionLocal() as db:
            existing = (
                db.query(SessionRecord).filter(SessionRecord.id == session_id).first()
            )

            if not existing:
                session_record = SessionRecord(id=session_id, provider=provider)
                db.add(session_record)
                db.commit()
                logger.info(f"Created new session: {session_id}")

    def store_thought(
        self,
        session_id: str,
        thought_data: ThoughtData,
        response: str | None = None,
        processing_metadata: dict | None = None,
    ) -> int:
        """Store a thought and return its database ID."""
        with self.SessionLocal() as db:
            session_record = self._ensure_session_exists(db, session_id)
            thought_record = self._create_thought_record(
                session_id, thought_data, response, processing_metadata
            )

            db.add(thought_record)
            self._update_session_stats(session_record, processing_metadata)
            self._handle_branching(db, session_id, thought_data)

            db.commit()
            return int(thought_record.id)

    def _ensure_session_exists(self, db: Session, session_id: str) -> SessionRecord:
        """Ensure session exists in database and return it."""
        session_record = (
            db.query(SessionRecord).filter(SessionRecord.id == session_id).first()
        )

        if not session_record:
            self.create_session(session_id)
            session_record = (
                db.query(SessionRecord).filter(SessionRecord.id == session_id).first()
            )

        return session_record

    def _create_thought_record(
        self,
        session_id: str,
        thought_data: ThoughtData,
        response: str | None,
        processing_metadata: dict | None,
    ) -> ThoughtRecord:
        """Create a thought record with metadata."""
        thought_record = ThoughtRecord(
            session_id=session_id,
            thought_number=thought_data.thoughtNumber,
            thought=thought_data.thought,
            total_thoughts=thought_data.totalThoughts,
            next_needed=thought_data.nextThoughtNeeded,
            branch_from=thought_data.branchFromThought,
            branch_id=thought_data.branchId,
            response=response,
        )

        if processing_metadata:
            self._apply_processing_metadata(thought_record, processing_metadata)

        return thought_record

    def _apply_processing_metadata(
        self, thought_record: ThoughtRecord, metadata: dict
    ) -> None:
        """Apply processing metadata to thought record."""
        if strategy := metadata.get("strategy"):
            thought_record.processing_strategy = str(strategy)  # type: ignore[assignment]
        if complexity_score := metadata.get("complexity_score"):
            thought_record.complexity_score = float(complexity_score)  # type: ignore[assignment]
        if estimated_cost := metadata.get("estimated_cost"):
            thought_record.estimated_cost = float(estimated_cost)  # type: ignore[assignment]
        if actual_cost := metadata.get("actual_cost"):
            thought_record.actual_cost = float(actual_cost)  # type: ignore[assignment]
        if token_usage := metadata.get("token_usage"):
            thought_record.token_usage = int(token_usage)  # type: ignore[assignment]
        if processing_time := metadata.get("processing_time"):
            thought_record.processing_time = float(processing_time)  # type: ignore[assignment]

        thought_record.specialist_used = metadata.get("specialists", [])
        thought_record.processed_at = datetime.utcnow()  # type: ignore[assignment]

    def _update_session_stats(
        self, session_record: SessionRecord, processing_metadata: dict | None
    ) -> None:
        """Update session statistics."""
        if session_record:
            session_record.totalThoughts += 1  # type: ignore[assignment]
            session_record.updated_at = datetime.utcnow()  # type: ignore[assignment]
            if processing_metadata and processing_metadata.get("actual_cost"):
                session_record.total_cost += float(processing_metadata["actual_cost"])  # type: ignore[assignment]

    def _handle_branching(
        self, db: Session, session_id: str, thought_data: ThoughtData
    ) -> None:
        """Handle branch record creation and updates."""
        if not thought_data.branchId:
            return

        branch_record = (
            db.query(BranchRecord)
            .filter(
                BranchRecord.session_id == session_id,
                BranchRecord.branchId == thought_data.branchId,
            )
            .first()
        )

        if not branch_record:
            branch_record = BranchRecord(
                session_id=session_id,
                branch_id=thought_data.branchId,
                parent_thought=thought_data.branchFromThought,
            )
            db.add(branch_record)

        branch_record.thought_count += 1  # type: ignore[assignment]

    def get_session_thoughts(
        self, session_id: str, limit: int | None = None
    ) -> list[ThoughtRecord]:
        """Retrieve thoughts for a session."""
        with self.SessionLocal() as db:
            query = (
                db.query(ThoughtRecord)
                .filter(ThoughtRecord.session_id == session_id)
                .order_by(ThoughtRecord.thoughtNumber)
            )

            if limit:
                query = query.limit(limit)

            return query.all()

    def get_thought_by_number(
        self, session_id: str, thought_number: int
    ) -> ThoughtRecord | None:
        """Get a specific thought by number."""
        with self.SessionLocal() as db:
            return (
                db.query(ThoughtRecord)
                .filter(
                    ThoughtRecord.session_id == session_id,
                    ThoughtRecord.thoughtNumber == thought_number,
                )
                .first()
            )

    def get_branch_thoughts(
        self, session_id: str, branch_id: str
    ) -> list[ThoughtRecord]:
        """Get all thoughts in a specific branch."""
        with self.SessionLocal() as db:
            return (
                db.query(ThoughtRecord)
                .filter(
                    ThoughtRecord.session_id == session_id,
                    ThoughtRecord.branchId == branch_id,
                )
                .order_by(ThoughtRecord.thoughtNumber)
                .all()
            )

    def prune_old_sessions(
        self, older_than_days: int = 30, keep_recent: int = 100
    ) -> int:
        """Prune old sessions to manage storage space."""
        cutoff_date = datetime.utcnow() - timedelta(days=older_than_days)

        with self.SessionLocal() as db:
            # Get sessions older than cutoff, excluding most recent ones
            old_sessions = (
                db.query(SessionRecord)
                .filter(SessionRecord.updated_at < cutoff_date)
                .order_by(desc(SessionRecord.updated_at))
                .offset(keep_recent)
            )

            deleted_count = 0
            for session in old_sessions:
                db.delete(session)  # Cascade will handle thoughts and branches
                deleted_count += 1

            if deleted_count > 0:
                db.commit()
                logger.info(f"Pruned {deleted_count} old sessions")

            return deleted_count

    def get_usage_stats(self, days_back: int = 7) -> dict:
        """Get usage statistics for cost optimization."""
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)

        with self.SessionLocal() as db:
            # Session stats
            session_count = (
                db.query(SessionRecord)
                .filter(SessionRecord.created_at >= cutoff_date)
                .count()
            )

            # Thought stats
            thought_stats = (
                db.query(ThoughtRecord)
                .filter(ThoughtRecord.created_at >= cutoff_date)
                .all()
            )

            total_thoughts = len(thought_stats)
            # Explicit type casting to resolve SQLAlchemy Column type issues
            total_cost: float = float(
                sum(float(t.actual_cost or 0) for t in thought_stats)
            )
            total_tokens: int = int(sum(int(t.token_usage or 0) for t in thought_stats))
            avg_processing_time = sum(
                t.processing_time or 0 for t in thought_stats
            ) / max(total_thoughts, 1)

            # Strategy breakdown
            strategy_stats = {}
            for thought in thought_stats:
                strategy = thought.processing_strategy or "unknown"
                if strategy not in strategy_stats:
                    strategy_stats[strategy] = {"count": 0, "cost": 0, "tokens": 0}
                strategy_stats[strategy]["count"] += 1
                strategy_stats[strategy]["cost"] += thought.actual_cost or 0
                strategy_stats[strategy]["tokens"] += thought.token_usage or 0

            return {
                "period_days": days_back,
                "session_count": session_count,
                "total_thoughts": total_thoughts,
                "total_cost": total_cost,
                "total_tokens": total_tokens,
                "avg_cost_per_thought": total_cost / max(total_thoughts, 1),
                "avg_tokens_per_thought": total_tokens / max(total_thoughts, 1),
                "avg_processing_time": avg_processing_time,
                "strategy_breakdown": strategy_stats,
            }

    def record_usage_metrics(
        self,
        provider: str,
        processing_strategy: str,
        complexity_level: str,
        thought_count: int = 1,
        tokens: int = 0,
        cost: float = 0.0,
        processing_time: float = 0.0,
        success: bool = True,
    ) -> None:
        """Record usage metrics for cost optimization."""
        with self.SessionLocal() as db:
            # Check if we have a record for today
            today = datetime.utcnow().date()
            existing = (
                db.query(UsageMetrics)
                .filter(
                    UsageMetrics.date >= datetime.combine(today, datetime.min.time()),
                    UsageMetrics.provider == provider,
                    UsageMetrics.processing_strategy == processing_strategy,
                    UsageMetrics.complexity_level == complexity_level,
                )
                .first()
            )

            if existing:
                # Update existing record
                existing.thought_count += thought_count  # type: ignore[assignment]
                existing.total_tokens += tokens  # type: ignore[assignment]
                existing.total_cost += cost  # type: ignore[assignment]
                existing.avg_processing_time = (  # type: ignore[assignment]
                    float(existing.avg_processing_time)
                    * (existing.thought_count - thought_count)
                    + processing_time * thought_count
                ) / existing.thought_count
                if not success:
                    existing.success_rate = (  # type: ignore[assignment]
                        float(existing.success_rate)
                        * (existing.thought_count - thought_count)
                    ) / existing.thought_count
            else:
                # Create new record
                metrics = UsageMetrics(
                    provider=provider,
                    processing_strategy=processing_strategy,
                    complexity_level=complexity_level,
                    thought_count=thought_count,
                    total_tokens=tokens,
                    total_cost=cost,
                    avg_processing_time=processing_time,
                    success_rate=1.0 if success else 0.0,
                )
                db.add(metrics)

            db.commit()

    def optimize_database(self) -> None:
        """Run database optimization tasks."""
        with self.SessionLocal() as db:
            from sqlalchemy import text

            if self.engine.dialect.name == "sqlite":
                db.execute(text("VACUUM"))
                db.execute(text("ANALYZE"))
            else:
                db.execute(text("ANALYZE"))
            db.commit()
            logger.info("Database optimization completed")

    def close(self) -> None:
        """Close database connections."""
        self.engine.dispose()


# Convenience functions
def create_persistent_memory(
    database_url: str | None = None,
) -> PersistentMemoryManager:
    """Create a persistent memory manager instance."""
    return PersistentMemoryManager(database_url)


def get_database_url_from_env() -> str:
    """Get database URL from environment variables."""
    if url := os.getenv("DATABASE_URL"):
        return url

    # Default to local SQLite
    db_dir = Path.home() / ".sequential_thinking" / "data"
    db_dir.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{db_dir}/memory.db"
