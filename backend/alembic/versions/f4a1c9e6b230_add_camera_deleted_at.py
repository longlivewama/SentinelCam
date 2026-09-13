"""add cameras.deleted_at

Revision ID: f4a1c9e6b230
Revises: 9973bfd223ea
Create Date: 2026-09-13 19:00:00.000000

Separates "camera deleted" from "camera turned off". Before this,
DELETE /api/cameras/{id} and a user-facing on/off toggle would have had to
share the same is_active flag, so switching a camera off would have made
it indistinguishable from a deleted one (both drop out of the camera
list). Nullable and unindexed: it is only ever read as IS NULL / IS NOT
NULL on a table sized in the tens to low hundreds of rows, not filtered
at scale.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision: str = 'f4a1c9e6b230'
down_revision: Union[str, Sequence[str], None] = '9973bfd223ea'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('cameras', sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('cameras', 'deleted_at')
