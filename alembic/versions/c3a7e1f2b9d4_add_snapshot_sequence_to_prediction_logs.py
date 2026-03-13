"""add snapshot_sequence to prediction logs

Revision ID: c3a7e1f2b9d4
Revises: b2f1c9d3a4e8
Create Date: 2026-03-13

Adds the snapshot_sequence column to tilt_prediction_logs.
This stores the full game trajectory of feature vectors (one 26-dim vector
per 5-second poll) as a JSON string. Used as training data for the GRU
temporal model once enough games are collected.

Shape at training time: (T, 26) where T = number of polling intervals.
A 40-minute game yields ~480 snapshots (~50KB per player).
"""
from alembic import op
import sqlalchemy as sa

revision = "c3a7e1f2b9d4"
down_revision = "b2f1c9d3a4e8"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "tilt_prediction_logs",
        sa.Column("snapshot_sequence", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("tilt_prediction_logs", "snapshot_sequence")
