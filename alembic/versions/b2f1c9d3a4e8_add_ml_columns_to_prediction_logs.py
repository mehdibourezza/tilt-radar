"""add ml columns to prediction logs

Revision ID: b2f1c9d3a4e8
Revises: e6848451e0af
Create Date: 2026-03-13

Adds four columns to tilt_prediction_logs to support the ML training pipeline:
  - role:                   player position (TOP/JUNGLE/MIDDLE/BOTTOM/UTILITY)
  - game_time_at_peak:      seconds into game when peak tilt score was reached
  - feature_vector_at_peak: 27-dim FeatureExtractor output as JSON array
  - n_signals_active:       number of signals active at peak (for fast queries)

All columns are nullable so existing records are unaffected.
New records written by the updated ws.py will populate these columns.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSON

# revision identifiers
revision = "b2f1c9d3a4e8"
down_revision = "e6848451e0af"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "tilt_prediction_logs",
        sa.Column("role", sa.String(16), nullable=True),
    )
    op.add_column(
        "tilt_prediction_logs",
        sa.Column("game_time_at_peak", sa.Float(), nullable=True),
    )
    op.add_column(
        "tilt_prediction_logs",
        sa.Column("feature_vector_at_peak", JSON, nullable=True),
    )
    op.add_column(
        "tilt_prediction_logs",
        sa.Column("n_signals_active", sa.Integer(), nullable=True),
    )

    # Index on role for role-stratified queries in calibration study
    op.create_index(
        "ix_tpl_role",
        "tilt_prediction_logs",
        ["role"],
    )


def downgrade() -> None:
    op.drop_index("ix_tpl_role", table_name="tilt_prediction_logs")
    op.drop_column("tilt_prediction_logs", "n_signals_active")
    op.drop_column("tilt_prediction_logs", "feature_vector_at_peak")
    op.drop_column("tilt_prediction_logs", "game_time_at_peak")
    op.drop_column("tilt_prediction_logs", "role")
