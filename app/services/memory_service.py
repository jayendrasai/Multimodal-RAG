from sqlalchemy import insert
from app.models.semantic_memory import SemanticMemory  # adjust to your actual model
from app.models.outbox import Outbox


async def store_semantic_fact(db_session, user_id: str, fact_text: str, embedding: list[float], metadata: dict):
    async with db_session.begin():
        result = await db_session.execute(
            insert(SemanticMemory)
            .values(user_id=user_id, fact_text=fact_text, metadata=metadata)
            .returning(SemanticMemory.id)
        )
        row_id = result.scalar_one()

        await db_session.execute(
            insert(Outbox).values(
                aggregate_type="semantic_memory",
                aggregate_id=row_id,
                payload={"embedding": embedding, "user_id": user_id, "metadata": metadata},
            )
        )
    # transaction commits here on context exit — both rows durable together or neither is