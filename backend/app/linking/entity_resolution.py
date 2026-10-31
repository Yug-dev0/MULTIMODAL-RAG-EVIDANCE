from app.extraction.embeddings import embed_texts
from app.linking.scoring_utils import cosine_similarity
from app.storage.repository import list_entities, persist_entity


async def resolve_entity(session, mention_text: str, mention_type: str, threshold: float = 0.85):
    normalized = mention_text.strip().lower()
    existing = await list_entities(session)
    for entity in existing:
        if normalized == entity.canonical_name.strip().lower():
            return entity
        if normalized in (a.strip().lower() for a in entity.aliases):
            return entity

    candidates_with_embedding = [e for e in existing if e.embedding is not None]
    if candidates_with_embedding:
        mention_embedding = (await embed_texts([mention_text]))[0]
        best_entity, best_similarity = None, 0.0
        for entity in candidates_with_embedding:
            similarity = cosine_similarity(mention_embedding, entity.embedding)
            if similarity > best_similarity:
                best_entity, best_similarity = entity, similarity
        if best_entity is not None and best_similarity >= threshold:
            if mention_text not in best_entity.aliases:
                best_entity.aliases = best_entity.aliases + [mention_text]
            return best_entity

    return await persist_entity(session, canonical_name=mention_text, entity_type=mention_type, aliases=[])
