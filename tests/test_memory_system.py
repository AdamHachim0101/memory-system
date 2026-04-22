import pytest
import asyncio
import uuid
import json
from datetime import datetime
from memory_system.services.postgres_service import PostgresService
from memory_system.services.redis_service import RedisService
from memory_system.services.neo4j_service import Neo4jService
from memory_system.services.embedding_service import NAGAEmbeddingService
from memory_system.services.memory_gateway import MemoryGateway
from memory_system.services.context_packer import pack_context
from memory_system.models import (
    IngestTurnRequest,
    RetrieveRequest,
    RetrievalProfile,
)


class TestMemorySystem:
    @pytest.fixture(autouse=True)
    async def setup_services(self):
        self.postgres = PostgresService()
        self.redis = RedisService()
        self.neo4j = Neo4jService()
        self.embedding = NAGAEmbeddingService()
        self.gateway = MemoryGateway(
            postgres=self.postgres,
            redis=self.redis,
            neo4j=self.neo4j,
            embedding_service=self.embedding,
        )

        await self.postgres.connect()
        await self.redis.connect()
        await self.neo4j.connect()
        await self.postgres.init_schema()

        yield

        await self.postgres.close()
        await self.redis.close()
        await self.neo4j.close()

    @pytest.fixture
    def test_context(self):
        return {
            "user_id": str(uuid.uuid4()),
            "agent_id": str(uuid.uuid4()),
            "conversation_id": str(uuid.uuid4()),
            "session_id": str(uuid.uuid4()),
        }

    async def test_single_turn_ingestion(self, test_context):
        request = IngestTurnRequest(
            user_id=test_context["user_id"],
            agent_id=test_context["agent_id"],
            conversation_id=test_context["conversation_id"],
            session_id=test_context["session_id"],
            turn_id=1,
            user_message="Hola, quiero información sobre pagos internacionales",
            assistant_message="Claro, puedo ayudarte con pagos internacionales. ¿Qué tipo de operación necesitas?",
            tool_events=[],
        )
        response = await self.gateway.ingest_turn(request)
        assert response.working_snapshot_id
        assert len(response.episodic_event_ids) >= 2
        assert response.candidate_count >= 0

    async def test_redis_cache_operations(self, test_context):
        session_id = test_context["session_id"]

        await self.redis.add_recent_turn(session_id, "user", "Hello world", 1)
        turns = await self.redis.get_recent_turns(session_id)
        assert len(turns) >= 1

        await self.redis.update_hot_entities(session_id, [{"type": "test", "value": "entity"}])
        entities = await self.redis.get_hot_entities(session_id)
        assert len(entities) >= 1

        await self.redis.cache_working_memory(session_id, {"objective": "test"})
        cached = await self.redis.get_cached_working_memory(session_id)
        assert cached is not None

        await self.redis.clear_session(session_id)

    async def test_postgres_working_memory(self, test_context):
        from memory_system.models import WorkingMemorySnapshot

        snapshot = WorkingMemorySnapshot(
            user_id=test_context["user_id"],
            agent_id=test_context["agent_id"],
            conversation_id=test_context["conversation_id"],
            session_id=test_context["session_id"],
            version=1,
            objective="Test objective",
            active_tasks=[{"task": "Test task", "status": "pending"}],
            constraints=["Constraint 1"],
            open_questions=["Question 1?"],
            active_entities=[{"type": "test", "value": "entity1"}],
            active_references=["ref1"],
            summary="Test summary",
        )
        snapshot_id = await self.postgres.save_working_memory(snapshot)
        assert snapshot_id

        retrieved = await self.postgres.get_latest_working_memory(test_context["conversation_id"])
        assert retrieved is not None
        assert retrieved.objective == "Test objective"

    async def test_postgres_episodic_events(self, test_context):
        from memory_system.models import EpisodicEvent, EventType

        event = EpisodicEvent(
            user_id=test_context["user_id"],
            agent_id=test_context["agent_id"],
            conversation_id=test_context["conversation_id"],
            session_id=test_context["session_id"],
            turn_id=1,
            role="user",
            event_type=EventType.MESSAGE.value,
            content="Test message content",
            salience_score=0.7,
        )
        event_id = await self.postgres.save_episodic_event(event)
        assert event_id

        events = await self.postgres.get_episodic_events(test_context["conversation_id"])
        assert len(events) >= 1

    async def test_retrieval_after_ingestion(self, test_context):
        request = IngestTurnRequest(
            user_id=test_context["user_id"],
            agent_id=test_context["agent_id"],
            conversation_id=test_context["conversation_id"],
            session_id=test_context["session_id"],
            turn_id=1,
            user_message="Quiero hacer un pago a Estados Unidos",
            assistant_message="Entendido. Para pagos a Estados Unidos necesito el monto y datos del beneficiario.",
            tool_events=[],
        )
        await self.gateway.ingest_turn(request)

        retrieve_request = RetrieveRequest(
            user_id=test_context["user_id"],
            agent_id=test_context["agent_id"],
            conversation_id=test_context["conversation_id"],
            session_id=test_context["session_id"],
            query="pago Estados Unidos",
            profile=RetrievalProfile.TASK,
        )
        response = await self.gateway.retrieve(retrieve_request)
        assert response is not None

    async def test_context_packer(self):
        working_memory = {
            "objective": "Process international payment",
            "active_tasks": [{"task": "Get beneficiary data", "status": "pending"}],
            "constraints": ["Must be under 10k"],
            "open_questions": ["Which bank?"],
        }
        semantic_memories = [
            {
                "memory_type": "user_preference",
                "canonical_text": "User prefers detailed explanations",
                "confidence": 0.8,
            }
        ]
        episodic_events = [
            {"role": "user", "content": "I need to send money to US", "turn_id": 1}
        ]

        packed = pack_context(
            working_memory=working_memory,
            semantic_memories=semantic_memories,
            episodic_events=episodic_events,
            digests=[],
            graph_context=[],
            profile=RetrievalProfile.TASK,
        )
        assert "Process international payment" in packed
        assert "User prefers detailed explanations" in packed

    async def test_memory_confidence_updates(self, test_context):
        from memory_system.models import SemanticMemory, MemoryType, StabilityClass, MemoryStatus

        memory = SemanticMemory(
            user_id=test_context["user_id"],
            agent_id=test_context["agent_id"],
            scope_type="conversation",
            scope_id=test_context["conversation_id"],
            memory_type=MemoryType.USER_PREFERENCE,
            subject="user",
            predicate="prefers",
            object_value="fast responses",
            canonical_text="User prefers fast responses",
            confidence=0.6,
            source_count=1,
            stability_class=StabilityClass.DURABLE,
            status=MemoryStatus.CANDIDATE,
        )
        memory_id = await self.postgres.save_semantic_memory(memory)
        assert memory_id

        await self.postgres.update_memory_confidence(memory_id, 0.8, increment_source=True)


class TestLongConversationMemory:
    @pytest.fixture(autouse=True)
    async def setup_services(self):
        self.postgres = PostgresService()
        self.redis = RedisService()
        self.neo4j = Neo4jService()
        self.embedding = NAGAEmbeddingService()
        self.gateway = MemoryGateway(
            postgres=self.postgres,
            redis=self.redis,
            neo4j=self.neo4j,
            embedding_service=self.embedding,
        )

        await self.postgres.connect()
        await self.redis.connect()
        await self.neo4j.connect()

        self.test_context = {
            "user_id": str(uuid.uuid4()),
            "agent_id": str(uuid.uuid4()),
            "conversation_id": str(uuid.uuid4()),
            "session_id": str(uuid.uuid4()),
        }

        yield

        await self.postgres.close()
        await self.redis.close()
        await self.neo4j.close()

    async def test_50_message_conversation(self):
        conversation_history = [
            {"turn": 1, "user": "Hola, necesito ayuda con un pago internacional", "assistant": "Claro, con gusto te ayudo. ¿Es un pago único o recurrente?"},
            {"turn": 2, "user": "Es único, necesito enviar dinero a Estados Unidos", "assistant": "Entendido. Para pagos a EE.UU. necesitas SWIFT/BIC del banco beneficiario. ¿Tienes esos datos?"},
            {"turn": 3, "user": "No todavía, el beneficiario me los va a enviar", "assistant": "Perfecto. Cuando los tengas, me los compartes y proceedemos con la operación."},
            {"turn": 4, "user": "Cuál es el proceso para enviar dinero a Asia?", "assistant": "Para Asia el proceso es similar pero puede requerir más documentos dependiendo del país destino."},
            {"turn": 5, "user": "Necesito saber las comisiones antes de proceder", "assistant": "Las comisiones varían según el monto y destino. Para EE.UU. typicalmente son $25-50 USD."},
            {"turn": 6, "user": "Hay alguna forma de reducir costos?", "assistant": "Sí, usar una cuenta en la misma moneda del destino puede reducir costos cambiarios."},
            {"turn": 7, "user": "Qué documentación necesito para pagos grandes?", "assistant": "Para montos acima de $10,000 USD necesitarás documentación adicional como factura o contrato."},
            {"turn": 8, "user": "Cuánto tiempo tarda un pago internacional?", "assistant": "Típicamente 2-5 días hábiles dependiendo del destino y bancos intermediarios."},
            {"turn": 9, "user": "Puedo rastrear el pago una vez enviado?", "assistant": "Sí, te proporciono un número de seguimiento once autorizado el pago."},
            {"turn": 10, "user": "Qué pasa si el pago falla?", "assistant": "Si falla, te notificamos inmediatamente y gestionamos la devolución sin costo adicional."},
            {"turn": 11, "user": "Necesito hacer un pago a Europa también", "assistant": "Para Europa necesitamos IBAN además del BIC. ¿Tienes ya los datos?"},
            {"turn": 12, "user": "Sí, tengo los datos del beneficiario europeo", "assistant": "Excelente. ¿Quieres proceder con el pago europeo o esperas primero el SWIFT de EE.UU.?"},
            {"turn": 13, "user": "Primero el de EE.UU., después vemos Europa", "assistant": "Entendido. Avísame cuando tengas los datos SWIFT para el pago a EE.UU."},
            {"turn": 14, "user": "El beneficiario me dijo que el SWIFT es CHASUS33", "assistant": "Perfecto, ese es el SWIFT de Chase Bank. Ahora necesito el número de cuenta del beneficiario."},
            {"turn": 15, "user": "El número de cuenta es 1234567890", "assistant": "Gracias. ¿El monto a enviar es el mismo que hablamos antes?"},
            {"turn": 16, "user": "Sí, son $5,000 USD", "assistant": "Con $5,000 USD las comisiones serán aproximadamente $30 USD más gastos de correspondencia."},
            {"turn": 17, "user": "Acepto las comisiones, puede proceder", "assistant": "Confirmado. Voy a procesar el pago de $5,000 USD a Chase Bank. Requiere confirmación adicionales?"},
            {"turn": 18, "user": "Sí, necesito confirmación por email", "assistant": "Te enviaré un email de confirmación con todos los detalles de la transacción."},
            {"turn": 19, "user": "También necesito el número de referencia del pago", "assistant": "Una vez procesado, te daré el número de referencia SWIFT (MT103) para seguimiento."},
            {"turn": 20, "user": "Qué es el MT103?", "assistant": "MT103 es el mensaje SWIFT estándar para pagos internacionales. Contiene todos los detalles de la transacción."},
            {"turn": 21, "user": "Puedo ver el MT103?", "assistant": "Sí, una vez enviado el pago te comparto el mensaje MT103 completo."},
            {"turn": 22, "user": "Cuánto tiempo para recibir confirmación?", "assistant": "La confirmación llega en 1-2 horas una vez procesado el pago."},
            {"turn": 23, "user": "Hay seguro contra fraudes?", "assistant": "Sí, tenemos protección de fraude incluida sin costo adicional para pagos internacionales."},
            {"turn": 24, "user": "Cómo reporto si hay problemas?", "assistant": "Puedes reportar problemas llamando al 800-XXX-XXXX o por email a fraude@banco.com."},
            {"turn": 25, "user": "También necesito información sobre pagos a México", "assistant": "Para México necesitas CLABE interbancaria de 18 dígitos. ¿Es para transferencia SPEI?"},
            {"turn": 26, "user": "Sí, es una transferencia SPEI", "assistant": "Para SPEI a México el proceso es más rápido, típicamente horas en lugar de días."},
            {"turn": 27, "user": "Cuál es el costo para México?", "assistant": "Para México el costo typical es $15 USD por transacción SPEI."},
            {"turn": 28, "user": "Hay límite de monto para SPEI?", "assistant": "No hay límite máximo, pero para montos acima de $10,000 USD requerimos documentación."},
            {"turn": 29, "user": "Qué documentación necesito para México?", "assistant": "Para México necesitas: CLABE, nombre completo del beneficiario y propósito de la transferencia."},
            {"turn": 30, "user": "El beneficiario es mi hermana, es para gastos familiares", "assistant": "Entendido. Para gastos familiares necesitas declaración de origen de fondos."},
            {"turn": 31, "user": "Dónde consigo ese formulario?", "assistant": "El formulario está disponible en nuestra web en la sección de descargas."},
            {"turn": 32, "user": "También necesito saber de políticas de cumplimiento", "assistant": "Para cumplimiento necesitamos verificar identidad, origen de fondos y propósito de cada transacción."},
            {"turn": 33, "user": "Cuánto tiempo guardan los registros?", "assistant": "Los registros se guardan por 5 años según regulaciones financieras."},
            {"turn": 34, "user": "Puedo hacer pagos en criptomonedas?", "assistant": "Actualmente no ofrecemos pagos en criptomonedas. Solo fiat (USD, EUR, MXN)."},
            {"turn": 35, "user": "Qué diferencia hay entre wire y SWIFT?", "assistant": "SWIFT es la red que procesa pagos internacionales. Wire es simplemente otro término para transferencia."},
            {"turn": 36, "user": "SWIFT es seguro?", "assistant": "SWIFT es extremadamente seguro con encriptación bancaria de grado militar."},
            {"turn": 37, "user": "Quién recibe mis datos bancarios?", "assistant": "Tus datos solo los recibe el banco beneficiario y nuestros sistemas cifrados."},
            {"turn": 38, "user": "Cómo sé que el pago llegó?", "assistant": "Te llega confirmación del banco beneficiario cuando acreditan los fondos."},
            {"turn": 39, "user": "Puedo cancelar un pago después de enviarlo?", "assistant": "Solo si aún no ha sido procesado. Una vez enviado no se puede cancelar."},
            {"turn": 40, "user": "Qué pasa si pongo mal los datos?", "assistant": "Si los datos no coinciden, el banco beneficiario devolverá los fondos. Por eso escritical verificar."},
            {"turn": 41, "user": "Cuánto tiempo para devolución si hay error?", "assistant": "La devolución typicalmente toma 5-10 días hábiles si los datos fueron incorrectos."},
            {"turn": 42, "user": "Hay algún costo por devolución?", "assistant": "Sí, hay costo de $25 USD por devolución más gastos de corresponsales."},
            {"turn": 43, "user": "Qué son los corresponsales?", "assistant": "Corresponsales son bancos intermediarios que procesan pagos entre bancos."},
            {"turn": 44, "user": "Por qué necesito saber de corresponsales?", "assistant": "Cada corresponsal cobra fees. Por eso pagos a ciudades pequeñas pueden costar más."},
            {"turn": 45, "user": "Puedes estimar el costo total para EE.UU.?", "assistant": "Para $5,000 a EE.UU. el costo total estimado es: comisión ours $30 + gastos correspondencia $15 = $45 USD."},
            {"turn": 46, "user": "El beneficiario recibe los $5,000 netos?", "assistant": "El beneficiario recibe $5,000 menos cualquier cargo del banco beneficiario en EE.UU."},
            {"turn": 47, "user": "Cuánto cobra Chase típicamente?", "assistant": "Chase típicamente cobra $15-25 USD por recibir transferencias internacionales."},
            {"turn": 48, "user": "Entonces mi hermana recibiría $4,975 aproximadamente?", "assistant": "Correcto. Recibiría aproximadamente $4,975-$4,980 USD neto."},
            {"turn": 49, "user": "Entonces debo告诉她她会收到大约这些钱", "assistant": "Sí,告诉她大约 $4,975-4,980 USD después de comisiones de Chase."},
            {"turn": 50, "user": "Gracias, guarda toda esta información para futuras referencias", "assistant": "Guardado. Tengo registrado: pagos EE.UU., México y Europa con comisiones y procesos. ¿Algo más?"},
        ]

        for msg in conversation_history:
            request = IngestTurnRequest(
                user_id=self.test_context["user_id"],
                agent_id=self.test_context["agent_id"],
                conversation_id=self.test_context["conversation_id"],
                session_id=self.test_context["session_id"],
                turn_id=msg["turn"],
                user_message=msg["user"],
                assistant_message=msg["assistant"],
                tool_events=[],
            )
            response = await self.gateway.ingest_turn(request)
            assert response.working_snapshot_id, f"Failed at turn {msg['turn']}"

        retrieve_request = RetrieveRequest(
            user_id=self.test_context["user_id"],
            agent_id=self.test_context["agent_id"],
            conversation_id=self.test_context["conversation_id"],
            session_id=self.test_context["session_id"],
            query="pagos internacionales Estados Unidos Mexico Europa",
            profile=RetrievalProfile.DEEP,
        )
        response = await self.gateway.retrieve(retrieve_request)

        assert len(response.episodic_events) > 0, "Should retrieve historical conversation events"
        assert len(response.semantic_memories) > 0 or len(response.episodic_events) > 0, "Should have memory or events"

        all_content = " ".join([e.content for e in response.episodic_events])
        assert "Estados Unidos" in all_content or "SWIFT" in all_content or "$5,000" in all_content, \
            f"Should recall key topics from conversation. Content: {all_content[:500]}"

    async def test_recall_specific_topic_from_history(self):
        test_context_specific = {
            "user_id": str(uuid.uuid4()),
            "agent_id": str(uuid.uuid4()),
            "conversation_id": str(uuid.uuid4()),
            "session_id": str(uuid.uuid4()),
        }

        conversation = [
            {"turn": 1, "user": "Estoy trabajando en un proyecto de arquitectura de microservicios", "assistant": "Interesante. ¿Qué tecnología estás usando para los microservicios?"},
            {"turn": 2, "user": "Estoy usando Python con FastAPI y Docker", "assistant": "Buena elección. FastAPI es muy performante. ¿Ya tienes el código en GitHub?"},
            {"turn": 3, "user": "Sí, está en github.com/mi-proyecto/microservices", "assistant": "Perfecto. ¿Necesitas ayuda con CI/CD también?"},
            {"turn": 4, "user": "Sí, quiero implementar GitHub Actions para despliegues automáticos", "assistant": "Para GitHub Actions necesitarás crear un workflow en .github/workflows/"},
            {"turn": 5, "user": "También estoy usando Redis para cache y Postgres para datos", "assistant": "Excelente stack. ¿Redis lo usas para sesión o para cache?"},
            {"turn": 6, "user": "Para ambos: sesión de usuarios y cache de consultas frecuentes", "assistant": "Buena estrategia. ¿Tienes métricas de hits en cache?"},
            {"turn": 7, "user": "No todavía, necesito implementar monitoring", "assistant": "Te recomiendo Prometheus + Grafana para métricas de cache y servicios."},
            {"turn": 8, "user": "Estoy en el proceso de configurar eso", "assistant": "¿Ya tienes Docker Compose para todo el entorno?"},
            {"turn": 9, "user": "Sí, tengo docker-compose.yml con todos los servicios", "assistant": "Perfecto. ¿El docker-compose incluye Redis, Postgres y Prometheus?"},
            {"turn": 10, "user": "Sí, todo está configurado. Ahora quiero conectar todo con Neo4j para GraphRAG", "assistant": "Excelente. Neo4j es great para GraphRAG. ¿Vas a usar el driver oficial de Neo4j para Python?"},
        ]

        for msg in conversation:
            request = IngestTurnRequest(
                user_id=test_context_specific["user_id"],
                agent_id=test_context_specific["agent_id"],
                conversation_id=test_context_specific["conversation_id"],
                session_id=test_context_specific["session_id"],
                turn_id=msg["turn"],
                user_message=msg["user"],
                assistant_message=msg["assistant"],
                tool_events=[],
            )
            await self.gateway.ingest_turn(request)

        retrieve_request = RetrieveRequest(
            user_id=test_context_specific["user_id"],
            agent_id=test_context_specific["agent_id"],
            conversation_id=test_context_specific["conversation_id"],
            session_id=test_context_specific["session_id"],
            query="microservicios Python FastAPI Docker Redis Postgres Neo4j",
            profile=RetrievalProfile.DEEP,
        )
        response = await self.gateway.retrieve(retrieve_request)

        all_content = " ".join([e.content for e in response.episodic_events])

        assert "microservicios" in all_content.lower() or "fastapi" in all_content.lower() or "docker" in all_content.lower(), \
            f"Should recall technical project details. Found content: {all_content[:300]}"
        assert "Redis" in all_content or "postgres" in all_content.lower() or "Neo4j" in all_content, \
            f"Should recall specific technologies mentioned. Content: {all_content[:300]}"

    async def test_memory_persistence_across_sessions(self):
        session1_context = {
            "user_id": str(uuid.uuid4()),
            "agent_id": str(uuid.uuid4()),
            "conversation_id": str(uuid.uuid4()),
            "session_id": str(uuid.uuid4()),
        }

        request1 = IngestTurnRequest(
            user_id=session1_context["user_id"],
            agent_id=session1_context["agent_id"],
            conversation_id=session1_context["conversation_id"],
            session_id=session1_context["session_id"],
            turn_id=1,
            user_message="Mi nombre es Carlos y prefiero respuestas técnicas detalladas",
            assistant_message="Hola Carlos, entendido. Daré respuestas técnicas detalladas.",
            tool_events=[],
        )
        await self.gateway.ingest_turn(request1)

        session2_context = {
            "user_id": session1_context["user_id"],
            "agent_id": str(uuid.uuid4()),
            "conversation_id": str(uuid.uuid4()),
            "session_id": str(uuid.uuid4()),
        }

        request2 = IngestTurnRequest(
            user_id=session2_context["user_id"],
            agent_id=session2_context["agent_id"],
            conversation_id=session2_context["conversation_id"],
            session_id=session2_context["session_id"],
            turn_id=1,
            user_message="Hola de nuevo, recordaras que mi nombre es Carlos",
            assistant_message="Hola Carlos, claro que sí. ¿En qué puedo ayudarte hoy?",
            tool_events=[],
        )
        await self.gateway.ingest_turn(request2)

        retrieve_request = RetrieveRequest(
            user_id=session2_context["user_id"],
            agent_id=session2_context["agent_id"],
            conversation_id=session2_context["conversation_id"],
            session_id=session2_context["session_id"],
            query="nombre Carlos prefiero respuestas técnicas",
            profile=RetrievalProfile.DEEP,
        )
        response = await self.gateway.retrieve(retrieve_request)

        all_semantic_text = " ".join([m.canonical_text for m in response.semantic_memories])
        all_episodic_text = " ".join([e.content for e in response.episodic_events])
        all_text = all_semantic_text + " " + all_episodic_text

        assert "Carlos" in all_text, f"Should remember user's name. Content: {all_text[:500]}"

    async def test_entity_tracking_across_conversation(self):
        test_context = {
            "user_id": str(uuid.uuid4()),
            "agent_id": str(uuid.uuid4()),
            "conversation_id": str(uuid.uuid4()),
            "session_id": str(uuid.uuid4()),
        }

        conversation = [
            {"turn": 1, "user": "Voy a trabajar con Amazon AWS para este proyecto", "assistant": "AWS es una excelente opción. ¿Qué servicios piensas usar?"},
            {"turn": 2, "user": "Piensa en EC2, S3 y Lambda", "assistant": "Buen stack serverless. Lambda se lleva bien con API Gateway."},
            {"turn": 3, "user": "También quiero integrar con Terraform para IaC", "assistant": "Terraform es great para IaC en AWS. ¿Ya tienes experiencia con Terraform?"},
            {"turn": 4, "user": "Sí, he trabajado con Terraform por 2 años", "assistant": "Perfecto. ¿Vas a usar módulos de Terraform para estructurar el código?"},
            {"turn": 5, "user": "Sí, usaré módulos para VPC, EC2 y RDS", "assistant": "Excelente. Para RDS necesitas crear subnet groups en la VPC."},
            {"turn": 6, "user": "También necesitaré configurar CloudWatch para monitoring", "assistant": "CloudWatch es nativo de AWS. ¿Necesitas dashboards personalizados?"},
            {"turn": 7, "user": "Sí, y también quiero alertas para costos", "assistant": "Para alertas de costos puedes usar AWS Budgets con CloudWatch."},
            {"turn": 8, "user": "Qué opinas de usar EKS en lugar de EC2?", "assistant": "EKS es mejor si necesitas Kubernetes. Si solo ejecutas funciones, Lambda es más económico."},
            {"turn": 9, "user": "Creo que Lambda es mejor para mi caso de uso", "assistant": "Lambda es ideal para cargas variables y pay-per-use."},
            {"turn": 10, "user": "Necesito saber más sobre API Gateway con Lambda", "assistant": "API Gateway puede invocar Lambda directamente. Soporta REST y WebSocket APIs."},
        ]

        for msg in conversation:
            request = IngestTurnRequest(
                user_id=test_context["user_id"],
                agent_id=test_context["agent_id"],
                conversation_id=test_context["conversation_id"],
                session_id=test_context["session_id"],
                turn_id=msg["turn"],
                user_message=msg["user"],
                assistant_message=msg["assistant"],
                tool_events=[],
            )
            await self.gateway.ingest_turn(request)

        retrieve_request = RetrieveRequest(
            user_id=test_context["user_id"],
            agent_id=test_context["agent_id"],
            conversation_id=test_context["conversation_id"],
            session_id=test_context["session_id"],
            query="AWS Lambda Terraform EKS API Gateway",
            profile=RetrievalProfile.DEEP,
            entities=["AWS", "Lambda", "Terraform", "EC2", "RDS", "CloudWatch"],
        )
        response = await self.gateway.retrieve(retrieve_request)

        all_content = " ".join([e.content for e in response.episodic_events])

        aws_mentioned = "AWS" in all_content or "Amazon" in all_content
        lambda_mentioned = "Lambda" in all_content
        terraform_mentioned = "Terraform" in all_content

        assert aws_mentioned and lambda_mentioned and terraform_mentioned, \
            f"Should recall multiple AWS technologies. AWS: {aws_mentioned}, Lambda: {lambda_mentioned}, Terraform: {terraform_mentioned}"

    async def test_context_packer_respects_profiles(self):
        working_memory = {
            "objective": "Test task",
            "active_tasks": [{"task": f"Task {i}", "status": "pending"} for i in range(20)],
            "constraints": [f"Constraint {i}" for i in range(10)],
            "open_questions": [f"Question {i}" for i in range(10)],
        }
        semantic_memories = [
            {"memory_type": "fact", "canonical_text": f"Fact {i}", "confidence": 0.8}
            for i in range(20)
        ]
        episodic_events = [
            {"role": "user", "content": f"Message {i}", "turn_id": i}
            for i in range(30)
        ]

        light_context = pack_context(
            working_memory, semantic_memories, episodic_events, [], [],
            RetrievalProfile.LIGHT
        )
        assert len(light_context) < 3000, f"Light profile should be concise, got {len(light_context)} chars"

        deep_context = pack_context(
            working_memory, semantic_memories, episodic_events, [], [],
            RetrievalProfile.DEEP
        )
        assert len(deep_context) > len(light_context), "Deep profile should have more content"