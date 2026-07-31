"""add_ingestion_outbox_columns

Revision ID: 910fcaf88606
Revises: 47c6cb1dd540
Create Date: 2026-07-13 11:58:58.005844

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '910fcaf88606'
down_revision: Union[str, None] = '47c6cb1dd540'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    op.add_column("outbox", sa.Column("document_id", sa.dialects.postgresql.UUID(as_uuid=True), nullable=True))
    op.create_index("idx_outbox_document_id", "outbox", ["document_id"])

    op.add_column(
        "outbox",
        sa.Column("operation_type", sa.Text(), nullable=False, server_default="UPSERT_MEMORY"),
    )
    op.execute(
        "ALTER TABLE outbox ADD CONSTRAINT outbox_operation_type_check "
        "CHECK (operation_type IN ('UPSERT_MEMORY','UPSERT_CHUNK','DELETE_CHUNKS'))"
    )

    #op.add_column("documents", sa.Column("chunk_count", sa.Integer, nullable=True))

    op.execute("ALTER TABLE documents DROP CONSTRAINT IF EXISTS documents_status_check")
    # op.execute(
    #     "ALTER TABLE documents ADD CONSTRAINT documents_status_check "
    #     "CHECK (status IN ('PROCESSING','READY','DELETING','DELETED','FAILED'))"
    # )
    op.execute(
        "ALTER TABLE documents ADD CONSTRAINT documents_status_check "
        "CHECK (status IN ('processing', 'ready', 'deleting', 'deleted', 'failed', 'PROCESSING', 'READY', 'DELETING', 'DELETED', 'FAILED'))"
    )


def downgrade():
    op.execute("ALTER TABLE documents DROP CONSTRAINT IF EXISTS documents_status_check")
    #op.drop_column("documents", "chunk_count")
    op.execute("ALTER TABLE outbox DROP CONSTRAINT IF EXISTS outbox_operation_type_check")
    op.drop_column("outbox", "operation_type")
    op.drop_index("idx_outbox_document_id", table_name="outbox")
    op.drop_column("outbox", "document_id")