"""initial baseline schema (users, cameras, events, recordings, password reset, video uploads)

Revision ID: dc14a9d062f3
Revises:
Create Date: 2026-09-06 20:34:17.482822

This is the project's first Alembic migration - previously the schema was
only ever created via `Base.metadata.create_all()` (see app/main.py),
never through a tracked migration. Rather than modeling this as an
incremental diff against that ad-hoc history (which would silently assume
the baseline tables already exist, and fail on any genuinely fresh
database - as it originally did before this fix), it creates the
complete current schema from scratch, matching app/models/*.py exactly.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'dc14a9d062f3'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table('cameras',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('name', sa.String(), nullable=False),
    sa.Column('url', sa.String(), nullable=False),
    sa.Column('camera_type', sa.String(), nullable=False),
    sa.Column('status', sa.String(), nullable=False),
    sa.Column('location', sa.String(), nullable=True),
    sa.Column('is_active', sa.Boolean(), nullable=False),
    sa.Column('ai_detection_enabled', sa.Boolean(), nullable=False),
    sa.Column('crowd_threshold', sa.Integer(), nullable=False),
    sa.Column('abandoned_object_seconds', sa.Integer(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_cameras_id'), 'cameras', ['id'], unique=False)
    op.create_table('users',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('email', sa.String(), nullable=False),
    sa.Column('password_hash', sa.String(), nullable=False),
    sa.Column('full_name', sa.String(), nullable=True),
    sa.Column('role', sa.String(), server_default='viewer', nullable=False),
    sa.Column('is_active', sa.Boolean(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)
    op.create_index(op.f('ix_users_id'), 'users', ['id'], unique=False)
    op.create_table('password_reset_tokens',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=False),
    sa.Column('token_hash', sa.String(), nullable=False),
    sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('used_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_password_reset_tokens_id'), 'password_reset_tokens', ['id'], unique=False)
    op.create_index(op.f('ix_password_reset_tokens_token_hash'), 'password_reset_tokens', ['token_hash'], unique=True)
    op.create_index(op.f('ix_password_reset_tokens_user_id'), 'password_reset_tokens', ['user_id'], unique=False)
    op.create_table('video_uploads',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=False),
    sa.Column('original_filename', sa.String(), nullable=False),
    sa.Column('stored_path', sa.String(), nullable=False),
    sa.Column('status', sa.String(), nullable=False),
    sa.Column('progress_percent', sa.Integer(), nullable=False),
    sa.Column('error_message', sa.String(), nullable=True),
    sa.Column('duration_seconds', sa.Float(), nullable=True),
    sa.Column('fps', sa.Float(), nullable=True),
    sa.Column('frame_count', sa.Integer(), nullable=True),
    sa.Column('persons_detected', sa.Integer(), nullable=False),
    sa.Column('fall_events_count', sa.Integer(), nullable=False),
    sa.Column('file_size_bytes', sa.Integer(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
    sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_video_uploads_id'), 'video_uploads', ['id'], unique=False)
    op.create_index(op.f('ix_video_uploads_user_id'), 'video_uploads', ['user_id'], unique=False)
    op.create_table('recordings',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('camera_id', sa.Integer(), nullable=True),
    sa.Column('video_upload_id', sa.Integer(), nullable=True),
    sa.Column('filename', sa.String(), nullable=False),
    sa.Column('file_path', sa.String(), nullable=False),
    sa.Column('duration_seconds', sa.Float(), nullable=False),
    sa.Column('trigger_action', sa.String(), nullable=False),
    sa.Column('file_size_bytes', sa.Integer(), nullable=False),
    sa.Column('event_timestamp', sa.DateTime(timezone=True), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
    sa.ForeignKeyConstraint(['camera_id'], ['cameras.id'], ),
    sa.ForeignKeyConstraint(['video_upload_id'], ['video_uploads.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_recordings_camera_id'), 'recordings', ['camera_id'], unique=False)
    op.create_index(op.f('ix_recordings_id'), 'recordings', ['id'], unique=False)
    op.create_index(op.f('ix_recordings_video_upload_id'), 'recordings', ['video_upload_id'], unique=False)
    op.create_table('events',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('camera_id', sa.Integer(), nullable=True),
    sa.Column('video_upload_id', sa.Integer(), nullable=True),
    sa.Column('recording_id', sa.Integer(), nullable=True),
    sa.Column('event_type', sa.String(), nullable=False),
    sa.Column('confidence_score', sa.Float(), nullable=False),
    sa.Column('timestamp', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
    sa.Column('triggered_recording', sa.Boolean(), nullable=False),
    sa.Column('acknowledged', sa.Boolean(), nullable=False),
    sa.Column('acknowledged_by', sa.Integer(), nullable=True),
    sa.Column('acknowledged_at', sa.DateTime(timezone=True), nullable=True),
    sa.ForeignKeyConstraint(['acknowledged_by'], ['users.id'], ),
    sa.ForeignKeyConstraint(['camera_id'], ['cameras.id'], ),
    sa.ForeignKeyConstraint(['recording_id'], ['recordings.id'], ),
    sa.ForeignKeyConstraint(['video_upload_id'], ['video_uploads.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_events_camera_id'), 'events', ['camera_id'], unique=False)
    op.create_index(op.f('ix_events_id'), 'events', ['id'], unique=False)
    op.create_index(op.f('ix_events_recording_id'), 'events', ['recording_id'], unique=False)
    op.create_index(op.f('ix_events_video_upload_id'), 'events', ['video_upload_id'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f('ix_events_video_upload_id'), table_name='events')
    op.drop_index(op.f('ix_events_recording_id'), table_name='events')
    op.drop_index(op.f('ix_events_id'), table_name='events')
    op.drop_index(op.f('ix_events_camera_id'), table_name='events')
    op.drop_table('events')
    op.drop_index(op.f('ix_recordings_video_upload_id'), table_name='recordings')
    op.drop_index(op.f('ix_recordings_id'), table_name='recordings')
    op.drop_index(op.f('ix_recordings_camera_id'), table_name='recordings')
    op.drop_table('recordings')
    op.drop_index(op.f('ix_video_uploads_user_id'), table_name='video_uploads')
    op.drop_index(op.f('ix_video_uploads_id'), table_name='video_uploads')
    op.drop_table('video_uploads')
    op.drop_index(op.f('ix_password_reset_tokens_user_id'), table_name='password_reset_tokens')
    op.drop_index(op.f('ix_password_reset_tokens_token_hash'), table_name='password_reset_tokens')
    op.drop_index(op.f('ix_password_reset_tokens_id'), table_name='password_reset_tokens')
    op.drop_table('password_reset_tokens')
    op.drop_index(op.f('ix_users_id'), table_name='users')
    op.drop_index(op.f('ix_users_email'), table_name='users')
    op.drop_table('users')
    op.drop_index(op.f('ix_cameras_id'), table_name='cameras')
    op.drop_table('cameras')
