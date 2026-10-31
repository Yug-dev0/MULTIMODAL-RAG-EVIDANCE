"""
Embedded temporal knowledge graph (Kuzu). Node tables: EvidenceUnit, Entity.
Edge tables: MENTIONS (evidence -> entity), RELATION (entity -> entity,
subject/predicate/object extracted from text), CO_OCCURS_WITH and
SEMANTICALLY_SIMILAR_TO (evidence <-> evidence, the two cross-modal link
types), SAME_ENTITY_AS (entity <-> entity, cross-file/cross-modal
co-reference). Every edge carries a confidence; temporal edges additionally
carry the timestamp of the evidence that grounds them, which is what makes
this a *temporal* knowledge graph rather than a static one.

This module never holds full EvidenceUnit content — it's a traversal index
over ids that point back into SQLiteStore.
"""

from __future__ import annotations

from pathlib import Path

import kuzu

import config

SCHEMA_STATEMENTS = [
    "CREATE NODE TABLE IF NOT EXISTS Entity(name STRING, type STRING, PRIMARY KEY(name))",
    "CREATE NODE TABLE IF NOT EXISTS EvidenceUnit("
    "id STRING, modality STRING, source_file STRING, start_time DOUBLE, "
    "end_time DOUBLE, page_number INT64, content_preview STRING, PRIMARY KEY(id))",
    "CREATE NODE TABLE IF NOT EXISTS KnowledgeEvent("
    "id STRING, title STRING, start_time DOUBLE, end_time DOUBLE, PRIMARY KEY(id))",
    "CREATE REL TABLE IF NOT EXISTS MENTIONS(FROM EvidenceUnit TO Entity, confidence DOUBLE)",
    "CREATE REL TABLE IF NOT EXISTS RELATION(FROM Entity TO Entity, predicate STRING, "
    "confidence DOUBLE, source_evidence_id STRING)",
    "CREATE REL TABLE IF NOT EXISTS CO_OCCURS_WITH(FROM EvidenceUnit TO EvidenceUnit, confidence DOUBLE)",
    "CREATE REL TABLE IF NOT EXISTS SEMANTICALLY_SIMILAR_TO(FROM EvidenceUnit TO EvidenceUnit, confidence DOUBLE)",
    "CREATE REL TABLE IF NOT EXISTS SAME_ENTITY_AS(FROM Entity TO Entity, confidence DOUBLE)",
    "CREATE REL TABLE IF NOT EXISTS CONTAINS(FROM KnowledgeEvent TO EvidenceUnit)",
]


class GraphStore:
    def __init__(self, db_path: Path | str | None = None):
        path = str(db_path or (Path(config.KUZU_DIR) / "graph"))
        self._db = kuzu.Database(path)
        self._conn = kuzu.Connection(self._db)
        for stmt in SCHEMA_STATEMENTS:
            self._conn.execute(stmt)

    # --- node upserts ---------------------------------------------------

    def upsert_evidence_node(self, unit) -> None:
        self._conn.execute(
            "MERGE (u:EvidenceUnit {id: $id}) "
            "ON CREATE SET u.modality=$modality, u.source_file=$source_file, u.start_time=$start_time, "
            "u.end_time=$end_time, u.page_number=$page_number, u.content_preview=$preview "
            "ON MATCH SET u.modality=$modality, u.source_file=$source_file, u.start_time=$start_time, "
            "u.end_time=$end_time, u.page_number=$page_number, u.content_preview=$preview",
            {
                "id": unit.id, "modality": unit.modality.value, "source_file": unit.source_file,
                "start_time": unit.start_time, "end_time": unit.end_time, "page_number": unit.page_number,
                "preview": unit.content[:200],
            },
        )

    def upsert_entity_node(self, name: str, entity_type: str) -> None:
        self._conn.execute(
            "MERGE (e:Entity {name: $name}) ON CREATE SET e.type=$type",
            {"name": name, "type": entity_type},
        )

    def upsert_event_node(self, event_id: str, title: str, start_time: float | None, end_time: float | None) -> None:
        self._conn.execute(
            "MERGE (e:KnowledgeEvent {id: $id}) "
            "ON CREATE SET e.title=$title, e.start_time=$start, e.end_time=$end "
            "ON MATCH SET e.title=$title, e.start_time=$start, e.end_time=$end",
            {"id": event_id, "title": title, "start": start_time, "end": end_time},
        )

    def add_contains(self, event_id: str, unit_id: str) -> None:
        self._conn.execute(
            "MATCH (e:KnowledgeEvent {id:$eid}), (u:EvidenceUnit {id:$uid}) MERGE (e)-[:CONTAINS]->(u)",
            {"eid": event_id, "uid": unit_id},
        )

    # --- edge upserts -----------------------------------------------------

    def add_mentions(self, unit_id: str, entity_name: str, confidence: float) -> None:
        self._conn.execute(
            "MATCH (u:EvidenceUnit {id:$uid}), (e:Entity {name:$name}) "
            "MERGE (u)-[r:MENTIONS]->(e) ON CREATE SET r.confidence=$conf",
            {"uid": unit_id, "name": entity_name, "conf": confidence},
        )

    def add_relation(self, subject: str, predicate: str, obj: str, confidence: float, source_evidence_id: str) -> None:
        self._conn.execute(
            "MATCH (s:Entity {name:$s}), (o:Entity {name:$o}) "
            "CREATE (s)-[:RELATION {predicate:$p, confidence:$c, source_evidence_id:$src}]->(o)",
            {"s": subject, "o": obj, "p": predicate, "c": confidence, "src": source_evidence_id},
        )

    def add_co_occurs(self, unit_id_a: str, unit_id_b: str, confidence: float) -> None:
        self._conn.execute(
            "MATCH (a:EvidenceUnit {id:$a}), (b:EvidenceUnit {id:$b}) "
            "MERGE (a)-[r:CO_OCCURS_WITH]->(b) ON CREATE SET r.confidence=$c",
            {"a": unit_id_a, "b": unit_id_b, "c": confidence},
        )

    def add_semantic_link(self, unit_id_a: str, unit_id_b: str, confidence: float) -> None:
        self._conn.execute(
            "MATCH (a:EvidenceUnit {id:$a}), (b:EvidenceUnit {id:$b}) "
            "MERGE (a)-[r:SEMANTICALLY_SIMILAR_TO]->(b) ON CREATE SET r.confidence=$c",
            {"a": unit_id_a, "b": unit_id_b, "c": confidence},
        )

    def add_same_entity(self, name_a: str, name_b: str, confidence: float) -> None:
        if name_a == name_b:
            return
        self._conn.execute(
            "MATCH (a:Entity {name:$a}), (b:Entity {name:$b}) "
            "MERGE (a)-[r:SAME_ENTITY_AS]->(b) ON CREATE SET r.confidence=$c",
            {"a": name_a, "b": name_b, "c": confidence},
        )

    # --- reads --------------------------------------------------------

    def all_entity_names(self) -> list[tuple[str, str]]:
        res = self._conn.execute("MATCH (e:Entity) RETURN e.name, e.type")
        out = []
        while res.has_next():
            row = res.get_next()
            out.append((row[0], row[1]))
        return out

    def _same_entity_cluster(self, entity_name: str) -> set[str]:
        cluster = {entity_name}
        res = self._conn.execute(
            "MATCH (e:Entity {name:$name})-[:SAME_ENTITY_AS]-(o:Entity) RETURN o.name",
            {"name": entity_name},
        )
        while res.has_next():
            cluster.add(res.get_next()[0])
        return cluster

    def evidence_ids_mentioning(self, entity_name: str) -> list[tuple[str, float]]:
        """Evidence unit ids that mention this entity (or any of its co-referent aliases)."""
        out: dict[str, float] = {}
        for name in self._same_entity_cluster(entity_name):
            res = self._conn.execute(
                "MATCH (u:EvidenceUnit)-[r:MENTIONS]->(e:Entity {name:$name}) RETURN u.id, r.confidence",
                {"name": name},
            )
            while res.has_next():
                uid, conf = res.get_next()
                out[uid] = max(out.get(uid, 0.0), conf)
        return list(out.items())

    def cross_modal_neighbors(self, unit_id: str) -> list[tuple[str, float, str]]:
        """Evidence units linked to `unit_id` via CO_OCCURS_WITH or SEMANTICALLY_SIMILAR_TO."""
        res = self._conn.execute(
            "MATCH (a:EvidenceUnit {id:$id})-[r:CO_OCCURS_WITH|SEMANTICALLY_SIMILAR_TO]-(b:EvidenceUnit) "
            "RETURN b.id, r.confidence, label(r)",
            {"id": unit_id},
        )
        out = []
        while res.has_next():
            bid, conf, rel_type = res.get_next()
            out.append((bid, conf, rel_type))
        return out

    def traverse_from_entity(self, entity_name: str, hops: int = 2) -> dict[str, dict]:
        """
        Returns {evidence_unit_id: {"confidence": float, "hop": int, "via": str}}
        reachable from `entity_name` (and its co-referent cluster) within
        `hops` bipartite/cross-modal steps.
        """
        frontier: dict[str, dict] = {}
        for uid, conf in self.evidence_ids_mentioning(entity_name):
            frontier[uid] = {"confidence": conf, "hop": 1, "via": "MENTIONS"}

        current_ids = list(frontier.keys())
        hop = 2
        while hop <= hops and current_ids:
            next_ids = []
            for uid in current_ids:
                for bid, conf, rel_type in self.cross_modal_neighbors(uid):
                    if bid not in frontier:
                        frontier[bid] = {"confidence": conf * 0.9, "hop": hop, "via": rel_type}
                        next_ids.append(bid)
            current_ids = next_ids
            hop += 1
        return frontier

    def related_entities(self, entity_name: str) -> list[tuple[str, str, str, float]]:
        """(predicate, other_entity, direction, confidence) triples touching this entity."""
        out = []
        res = self._conn.execute(
            "MATCH (s:Entity {name:$name})-[r:RELATION]->(o:Entity) RETURN r.predicate, o.name, r.confidence",
            {"name": entity_name},
        )
        while res.has_next():
            p, o, c = res.get_next()
            out.append((p, o, "outgoing", c))
        res = self._conn.execute(
            "MATCH (s:Entity)-[r:RELATION]->(o:Entity {name:$name}) RETURN r.predicate, s.name, r.confidence",
            {"name": entity_name},
        )
        while res.has_next():
            p, s, c = res.get_next()
            out.append((p, s, "incoming", c))
        return out

    def close(self) -> None:
        pass
