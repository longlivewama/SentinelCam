"""add events.video_timestamp_seconds and events.detector

Revision ID: b7f2a1c48e10
Revises: dc14a9d062f3
Create Date: 2026-09-08 17:05:00.000000

Two additive, nullable columns on `events`:

  video_timestamp_seconds  Where in an uploaded video the event was
                           detected (seconds from the start of the file).
                           The analysis pipeline already computed this per
                           fall but discarded it, leaving the UI to show
                           the wall-clock `timestamp` - which is when the
                           worker happened to reach that frame, not a
                           position in the footage.

  detector                 Which strategy fired the event ("model" for the
                           trained YOLO fall detector, "heuristic" for the
                           pose/geometry detectors), so a confidence score
                           can be interpreted correctly - the two are
                           computed differently and are not comparable.

Both are nullable with no backfill: existing rows genuinely do not have
this information, and inventing a value for them would be worse than
showing nothing.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b7f2a1c48e10'
down_revision: Union[str, Sequence[str], None] = 'dc14a9d062f3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('events', sa.Column('video_timestamp_seconds', sa.Float(), nullable=True))
    op.add_column('events', sa.Column('detector', sa.String(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('events', 'detector')
    op.drop_column('events', 'video_timestamp_seconds')
