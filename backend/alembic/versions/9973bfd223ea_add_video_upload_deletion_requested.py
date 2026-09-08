"""add video_uploads.deletion_requested

Revision ID: 9973bfd223ea
Revises: c3d81f0a55b2
Create Date: 2026-09-09 00:00:00.000000

Deleting an upload while its analysis worker is still running raced the
worker: the worker held the stored_path and could still be mid-decode
when the row (and file) vanished under it, and a fall detected in that
window could try to INSERT a Recording/Event referencing an
already-deleted video_upload_id, hitting the FK constraint from inside
the worker thread.

This column lets DELETE defer the actual removal instead of racing it:
while status is pending/processing, DELETE just flips this flag and
returns; the worker notices it (via a locked, throttled check) and
performs the real cascade-delete itself once it stops touching the row.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision: str = '9973bfd223ea'
down_revision: Union[str, Sequence[str], None] = 'c3d81f0a55b2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        'video_uploads',
        sa.Column('deletion_requested', sa.Boolean(), nullable=False, server_default='false'),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('video_uploads', 'deletion_requested')
