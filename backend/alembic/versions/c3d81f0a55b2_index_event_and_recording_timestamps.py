"""index events.timestamp and recordings.event_timestamp

Revision ID: c3d81f0a55b2
Revises: b7f2a1c48e10
Create Date: 2026-09-08 18:05:00.000000

Both list endpoints (`GET /api/alerts`, `GET /api/recordings`) order by
these columns newest-first, and the analytics summary filters events by a
time window. Neither column was indexed, so each of those queries is a
sequential scan plus a sort over a table that grows for as long as a
camera keeps running.
"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = 'c3d81f0a55b2'
down_revision: Union[str, Sequence[str], None] = 'b7f2a1c48e10'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_index('ix_events_timestamp', 'events', ['timestamp'])
    op.create_index('ix_recordings_event_timestamp', 'recordings', ['event_timestamp'])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_recordings_event_timestamp', table_name='recordings')
    op.drop_index('ix_events_timestamp', table_name='events')
