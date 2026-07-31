"""add_outbox_table

Revision ID: 47c6cb1dd540
Revises: 79a620516cab
Create Date: 2026-07-12 08:58:28.685056

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '47c6cb1dd540'
down_revision: Union[str, None] = '79a620516cab'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    op.create_table(
        "outbox",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("aggregate_type", sa.Text, nullable=False),
        sa.Column("aggregate_id", sa.dialects.postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("payload", sa.dialects.postgresql.JSONB, nullable=False),
        sa.Column("status", sa.Text, nullable=False, server_default="PENDING"),
        sa.Column("retry_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index(
        "idx_outbox_pending", "outbox", ["status"],
        postgresql_where=sa.text("status = 'PENDING'"),
    )
    op.execute(
        "ALTER TABLE outbox ADD CONSTRAINT outbox_status_check "
        "CHECK (status IN ('PENDING','PUBLISHED','DEAD'))"
    )

def downgrade():
    op.drop_index("idx_outbox_pending", table_name="outbox")
    op.drop_table("outbox")
