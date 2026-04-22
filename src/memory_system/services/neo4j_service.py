from neo4j import AsyncGraphDatabase
from typing import Optional
from memory_system.config import settings


class Neo4jService:
    def __init__(self):
        self.driver = None

    async def connect(self):
        self.driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )

    async def close(self):
        if self.driver:
            await self.driver.close()

    async def execute_query(self, cypher: str, params: dict = None) -> list[dict]:
        params = params or {}
        async with self.driver.session() as session:
            result = await session.run(cypher, params)
            records = await result.data()
            return records

    async def ensure_user_node(self, user_id: str) -> None:
        cypher = """
        MERGE (u:User {id: $user_id})
        ON CREATE SET u.created_at = datetime()
        RETURN u
        """
        await self.execute_query(cypher, {"user_id": user_id})

    async def ensure_conversation_node(
        self, conversation_id: str, user_id: str
    ) -> None:
        cypher = """
        MERGE (c:Conversation {id: $conversation_id})
        ON CREATE SET c.created_at = datetime()
        WITH c
        MATCH (u:User {id: $user_id})
        MERGE (u)-[:HAS_CONVERSATION]->(c)
        """
        await self.execute_query(cypher, {"conversation_id": conversation_id, "user_id": user_id})

    async def ensure_session_node(self, session_id: str, conversation_id: str) -> None:
        cypher = """
        MERGE (s:Session {id: $session_id})
        ON CREATE SET s.created_at = datetime()
        WITH s
        MATCH (c:Conversation {id: $conversation_id})
        MERGE (s)-[:BELONGS_TO]->(c)
        """
        await self.execute_query(cypher, {"session_id": session_id, "conversation_id": conversation_id})

    async def add_entity_mention(
        self, entity_type: str, entity_value: str, context: str, conversation_id: str
    ) -> None:
        cypher = """
        MERGE (e:Entity {type: $entity_type, value: $entity_value})
        ON CREATE SET e.created_at = datetime()
        WITH e
        MATCH (c:Conversation {id: $conversation_id})
        MERGE (e)-[:MENTIONED_IN {context: $context, at: datetime()}]->(c)
        """
        await self.execute_query(cypher, {
            "entity_type": entity_type,
            "entity_value": entity_value,
            "context": context,
            "conversation_id": conversation_id,
        })

    async def relate_entities(
        self, entity1_type: str, entity1_value: str,
        entity2_type: str, entity2_value: str,
        relation_type: str
    ) -> None:
        cypher = """
        MATCH (e1:Entity {type: $entity1_type, value: $entity1_value})
        MATCH (e2:Entity {type: $entity2_type, value: $entity2_value})
        MERGE (e1)-[r:RELATED_TO {type: $relation_type}]->(e2)
        SET r.created_at = datetime()
        """
        await self.execute_query(cypher, {
            "entity1_type": entity1_type,
            "entity1_value": entity1_value,
            "entity2_type": entity2_type,
            "entity2_value": entity2_value,
            "relation_type": relation_type,
        })

    async def add_decision(
        self, decision_id: str, decision_text: str, conversation_id: str
    ) -> None:
        cypher = """
        MERGE (d:Decision {id: $decision_id})
        SET d.text = $decision_text, d.created_at = datetime()
        WITH d
        MATCH (c:Conversation {id: $conversation_id})
        MERGE (d)-[:DECIDED_IN]->(c)
        """
        await self.execute_query(cypher, {
            "decision_id": decision_id,
            "decision_text": decision_text,
            "conversation_id": conversation_id,
        })

    async def link_memory_to_entity(
        self, memory_id: str, entity_type: str, entity_value: str
    ) -> None:
        cypher = """
        MATCH (m:Memory {id: $memory_id})
        MATCH (e:Entity {type: $entity_type, value: $entity_value})
        MERGE (m)-[:ABOUT]->(e)
        """
        await self.execute_query(cypher, {
            "memory_id": memory_id,
            "entity_type": entity_type,
            "entity_value": entity_value,
        })

    async def get_entity_context(
        self, entity_type: str, entity_value: str, depth: int = 2
    ) -> list[dict]:
        cypher = f"""
        MATCH (e:Entity {{type: $entity_type, value: $entity_value}})
        MATCH (e)-[r*1..{depth}]-(connected)
        RETURN e, r, connected
        LIMIT 50
        """
        return await self.execute_query(cypher, {"entity_type": entity_type, "entity_value": entity_value})

    async def get_timeline(
        self, conversation_id: str, limit: int = 50
    ) -> list[dict]:
        cypher = """
        MATCH (c:Conversation {id: $conversation_id})
        MATCH (c)<-[:DECIDED_IN]-(d:Decision)
        OPTIONAL MATCH (c)<-[:MENTIONED_IN]-(e:Entity)
        WITH c, d, e
        ORDER BY COALESCE(d.created_at, e.created_at) DESC
        LIMIT $limit
        RETURN c, d, e
        """
        return await self.execute_query(cypher, {"conversation_id": conversation_id, "limit": limit})

    async def get_related_entities(
        self, entity_type: str, entity_value: str, limit: int = 10
    ) -> list[dict]:
        cypher = """
        MATCH (e:Entity {type: $entity_type, value: $entity_value})
        MATCH (e)-[:RELATED_TO]-(other)
        RETURN other
        LIMIT $limit
        """
        return await self.execute_query(cypher, {"entity_type": entity_type, "entity_value": entity_value, "limit": limit})

    async def record_contradiction(
        self, memory_id_1: str, memory_id_2: str, notes: str = ""
    ) -> None:
        cypher = """
        MATCH (m1:Memory {id: $memory_id_1})
        MATCH (m2:Memory {id: $memory_id_2})
        MERGE (m1)-[r:CONTRADICTS]->(m2)
        SET r.notes = $notes, r.detected_at = datetime()
        """
        await self.execute_query(cypher, {
            "memory_id_1": memory_id_1,
            "memory_id_2": memory_id_2,
            "notes": notes,
        })

    async def get_memory_graph(
        self, user_id: str, memory_type: str = None, limit: int = 50
    ) -> list[dict]:
        if memory_type:
            cypher = """
            MATCH (u:User {id: $user_id})-[:HAS_CONVERSATION]->(c:Conversation)
            MATCH (m:Memory {type: $memory_type})-[:ABOUT]->(e:Entity)
            RETURN m, e
            LIMIT $limit
            """
            return await self.execute_query(cypher, {"user_id": user_id, "memory_type": memory_type, "limit": limit})
        else:
            cypher = """
            MATCH (u:User {id: $user_id})-[:HAS_CONVERSATION]->(c:Conversation)
            MATCH (m:Memory)-[:ABOUT]->(e:Entity)
            RETURN m, e
            LIMIT $limit
            """
            return await self.execute_query(cypher, {"user_id": user_id, "limit": limit})
