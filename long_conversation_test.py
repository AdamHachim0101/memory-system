#!/usr/bin/env python3
"""
Long Conversation Memory Test
Simulates a 50+ message conversation and verifies the memory system
can recall important details from earlier in the conversation.
"""
import asyncio
import uuid
from memory_system.services.postgres_service import PostgresService
from memory_system.services.redis_service import RedisService
from memory_system.services.neo4j_service import Neo4jService
from memory_system.services.embedding_service import NAGAEmbeddingService
from memory_system.services.memory_gateway import MemoryGateway
from memory_system.services.context_packer import pack_context
from memory_system.models import IngestTurnRequest, RetrieveRequest, RetrievalProfile


async def run_long_conversation_test():
    print("=" * 60)
    print("LONG CONVERSATION MEMORY TEST")
    print("=" * 60)

    postgres = PostgresService()
    redis = RedisService()
    neo4j = Neo4jService()
    embedding = NAGAEmbeddingService()
    gateway = MemoryGateway(postgres, redis, neo4j, embedding)

    await postgres.connect()
    await redis.connect()
    await neo4j.connect()
    await postgres.init_schema()

    test_context = {
        "user_id": str(uuid.uuid4()),
        "agent_id": str(uuid.uuid4()),
        "conversation_id": str(uuid.uuid4()),
        "session_id": str(uuid.uuid4()),
    }

    conversation = [
        {"turn": 1, "user": "Hola, soy Carlos y trabajo en el departamento de finanzas", "assistant": "Hola Carlos, bienvenido. ¿En qué puedo ayudarte hoy?"},
        {"turn": 2, "user": "Necesito hacer un pago internacional a Estados Unidos por $10,000", "assistant": "Entendido Carlos. Para pagos a EE.UU. necesitamos el SWIFT/BIC del banco beneficiario y sus datos completos."},
        {"turn": 3, "user": "El beneficiario es mi proveedor TechCorp Inc., el SWIFT es CHASUS33", "assistant": "Perfecto. ¿Tienes el número de cuenta del beneficiario?"},
        {"turn": 4, "user": "Sí, es 1234567890 en Chase Bank", "assistant": "¿Cuál es el propósito de este pago?"},
        {"turn": 5, "user": "Es para pagar servicios de software de su empresa", "assistant": "Para servicios necesitas factura o contrato. ¿Tienes alguno?"},
        {"turn": 6, "user": "Sí, tengo el contrato digital firmado", "assistant": "¿Podrías compartirlo para procesarlo junto con el pago?"},
        {"turn": 7, "user": "Lo enviaré por email después", "assistant": "Bien. Una vez queverifiquemos los documentos, procesaremos el pago."},
        {"turn": 8, "user": "¿Cuánto tiempo tarda el pago?", "assistant": "Típicamente 2-3 días hábiles a EE.UU."},
        {"turn": 9, "user": "¿Cuáles son las comisiones?", "assistant": "Para EE.UU. la comisión ours es $25 más gastos de corresponsales que pueden ser $15-25."},
        {"turn": 10, "user": "¿Podemos reducir costos de alguna forma?", "assistant": "Sí, usando una cuenta en USD para evitar conversión de moneda."},
        {"turn": 11, "user": "También necesito hacer un pago a Europa pronto", "assistant": "Para Europa necesitamos IBAN además del BIC. ¿Sabes los datos?"},
        {"turn": 12, "user": "Mi proveedor europeo me los va a enviar la próxima semana", "assistant": "Entonces agendamos ese pago para después. Te avisamos cuando esté listo."},
        {"turn": 13, "user": "También quería preguntar sobre inversiones", "assistant": "Claro Carlos, tenemos varias opciones de inversión. ¿Te interesa algo específico?"},
        {"turn": 14, "user": "Algo conservador, para el dinero que no uso inmediatamente", "assistant": "Para perfil conservador recomendamos fondos del mercado monetario o bonos de gobierno."},
        {"turn": 15, "user": "¿Qué rendimiento anual tendría?", "assistant": "Los fondos monetarios están dando entre 4-5% APY actualmente."},
        {"turn": 16, "user": "¿Hay alguna penalización por retiro temprano?", "assistant": "Los fondos monetarios típicamente no tienen penalización, puedes retirar cuando quieras."},
        {"turn": 17, "user": "¿Cuánto mínimo puedo invertir?", "assistant": "El mínimo para fondos monetarios es $1,000 USD."},
        {"turn": 18, "user": "Perfecto, voy a transferir $5,000 de mi cuenta de cheques", "assistant": "Puedo ayudarte con esa transferencia. ¿Quieres que prepare la instrucción?"},
        {"turn": 19, "user": "Sí por favor, y también necesito ver el estado de mi cuenta", "assistant": "Tu cuenta tiene disponible $45,000 USD y $12,000 EUR."},
        {"turn": 20, "user": "¿Puedo ver los movimientos de los últimos 3 meses?", "assistant": "Sí, te muestro los movimientos recientes. Hubo 15 transacciones en ese período."},
        {"turn": 21, "user": "¿Hubo algún cargo inusual?", "assistant": "No hay cargos inusuales. Todos son transacciones reconocidas."},
        {"turn": 22, "user": "También necesito una tarjeta de crédito adicional para mi asistente", "assistant": "Podemos agregar una tarjeta adicional. ¿Para quién es?"},
        {"turn": 23, "user": "Para María, mi asistente executive", "assistant": "¿Cuál es el límite que necesitas para la tarjeta de María?"},
        {"turn": 24, "user": "$2,000 debería ser suficiente", "assistant": "Procesaremos la tarjeta con ese límite. ¿Dirección de facturación?"},
        {"turn": 25, "user": "La misma que la mía, en la oficina central", "assistant": "Entendido. La tarjeta llegará en 5-7 días hábiles."},
        {"turn": 26, "user": "¿Hay algún cargo anual?", "assistant": "La tarjeta adicional tiene $0 annual fee para cuentas premium."},
        {"turn": 27, "user": "Muy bien. ¿Podemos agendar una llamada para la próxima semana?", "assistant": "Tienes disponible el martes o jueves de 10am a 3pm. ¿Qué día te conviene?"},
        {"turn": 28, "user": "Martes a las 11am por favor", "assistant": "Confirmado. Te enviare la invitación por email."},
        {"turn": 29, "user": "Gracias. Ahora sobre el pago a EE.UU., ¿pueden acelerarlo?", "assistant": "Sí, podemos hacer un pago urgente por $15 adicionales. ¿Lo confirmas?"},
        {"turn": 30, "user": "Sí, confirmado. El pago es urgente", "assistant": "Entendido. Lo procesamos hoy mismo con el cargo urgente."},
        {"turn": 31, "user": "¿Puedo recibir confirmación por SMS?", "assistant": "Sí, podemos enviar SMS al número que tienes registrado."},
        {"turn": 32, "user": "Perfecto. También necesito saber si ofrecen préstamos comerciales", "assistant": "Sí, ofrecemos préstamos comerciales con tasas desde 6.99% APY."},
        {"turn": 33, "user": "¿Cuál es el proceso y requisitos?", "assistant": "Necesitamos estados financieros, plan de negocios y antigüedad mínima de 2 años."},
        {"turn": 34, "user": "Tenemos 3 años operando, ¿calificamos?", "assistant": "Sí, con 3 años de antigüedad calificas. ¿Cuánto necesitas?"},
        {"turn": 35, "user": "Estamos pensando en $50,000 para expansión", "assistant": "Podemos estructurar un préstamo de $50,000 a 5 años con tasa fija."},
        {"turn": 36, "user": "¿Qué documentación necesitan?", "assistant": "Estados financieros de los últimos 2 años,计划 de negocio y extracts bancarios."},
        {"turn": 37, "user": "Los tengo digitalizados, los envío esta semana", "assistant": "Bien. Una vez recibidos, la aprobación tarda 3-5 días hábiles."},
        {"turn": 38, "user": "¿Hay alguna forma de accelerate el proceso?", "assistant": "Si eres cliente premium podemos acelerado a 24-48 horas."},
        {"turn": 39, "user": "Sí, somos clientes premium", "assistant": "Perfecto. Entonces puedes acceder al proceso acelerado."},
        {"turn": 40, "user": "Excelente. ¿Qué otras ventajas tiene ser premium?", "assistant": "Como premium tienes: tasas preferenciales, tarjetas sin anualidad, manager dedicado y acceso a investimentos exclusivos."},
        {"turn": 41, "user": "¿Podemos hablar de optimización fiscal?", "assistant": "Tenemos servicios de consultoría fiscal. ¿Es para tu negocio o personal?"},
        {"turn": 42, "user": "Para TechCorp, mi empresa", "assistant": "Para empresas tenemos planes de optimización que pueden ahorrar 15-20% en impuestos."},
        {"turn": 43, "user": "¿Cuánto cuesta el servicio?", "assistant": "El servicio básico es $500 mensual, el completo es $1,200 mensual."},
        {"turn": 44, "user": "Empezamos con el básico por 3 meses a ver resultados", "assistant": "Entendido. Te conectamos con nuestro equipo fiscal."},
        {"turn": 45, "user": "También necesito capacitación para mi equipo en herramientas digitales", "assistant": "Ofrecemos webinars mensuales y cursos en línea gratuitos para clientes."},
        {"turn": 46, "user": "¿Pueden certificar al equipo?", "assistant": "Sí, tenemos programa de certificación en banking digital."},
        {"turn": 47, "user": "¿Cuántas personas pueden participar?", "assistant": "Hasta 10 personas del equipo sin costo adicional."},
        {"turn": 48, "user": "Perfecto, inscribiré a mi equipo de 8 personas", "assistant": "Te inscribo. Recibirás un email con las credenciales de acceso."},
        {"turn": 49, "user": "Gracias por todo. Guarda todas estas solicitudes para referencia futura", "assistant": "Guardado. Quedan registradas: pago EE.UU. urgente, préstamo $50K, tarjeta adicional María, capacitación equipo y consultoría fiscal básica."},
        {"turn": 50, "user": "Sí, por favor ten todo esto presente para nuestras conversaciones futuras", "assistant": "Entendido Carlos. Todo queda registrado en tu perfil. ¿Hay algo más?"},
    ]

    print(f"\nIngesting {len(conversation)} conversation turns...")
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
        response = await gateway.ingest_turn(request)
        if msg["turn"] % 10 == 0:
            print(f"  Turn {msg['turn']} processed (snapshot: {response.working_snapshot_id[:8]}...)")

    print("\n" + "-" * 60)
    print("RETRIEVAL TESTS")
    print("-" * 60)

    recall_tests = [
        {
            "name": "User's name",
            "query": "Carlos",
            "expected": ["Carlos"],
        },
        {
            "name": "Payment to USA",
            "query": "Estados Unidos TechCorp",
            "expected": ["Estados Unidos", "TechCorp", "$10,000"],
        },
        {
            "name": "European payment",
            "query": "Europa IBAN",
            "expected": ["Europa", "IBAN", "próxima semana"],
        },
        {
            "name": "Investment interest",
            "query": "inversión conservador fondo monetario",
            "expected": ["inversión", "conservador", "fondo"],
        },
        {
            "name": "Credit card for assistant",
            "query": "tarjeta adicional María asistente",
            "expected": ["tarjeta", "María"],
        },
        {
            "name": "Business loan",
            "query": "préstamo $50,000 expansión",
            "expected": ["préstamo", "$50,000", "expansión"],
        },
        {
            "name": "Premium benefits",
            "query": "cliente premium ventajas",
            "expected": ["premium", "tasas", "dedicado"],
        },
        {
            "name": "Tax consulting",
            "query": "optimización fiscal TechCorp",
            "expected": ["fiscal", "TechCorp"],
        },
        {
            "name": "Team training",
            "query": "capacitación equipo webinar",
            "expected": ["equipo", "webinar", "certificación"],
        },
    ]

    all_passed = True
    for test in recall_tests:
        retrieve_request = RetrieveRequest(
            user_id=test_context["user_id"],
            agent_id=test_context["agent_id"],
            conversation_id=test_context["conversation_id"],
            session_id=test_context["session_id"],
            query=test["query"],
            profile=RetrievalProfile.DEEP,
        )
        response = await gateway.retrieve(retrieve_request)

        all_content = " ".join([e.content for e in response.episodic_events]).lower()

        simple_query = test["query"].split()[0]
        direct_match = simple_query.lower() in all_content
        found_keywords = [kw for kw in test["expected"] if kw.lower() in all_content]

        status = "PASS" if (len(found_keywords) >= 2 or direct_match) else "FAIL"
        if status == "FAIL":
            all_passed = False
        print(f"\n[{status}] {test['name']}")
        print(f"  Query: {test['query']}")
        print(f"  Direct match '{simple_query}': {direct_match}")
        print(f"  Found keywords: {found_keywords if found_keywords else 'NOTHING RELEVANT'}")
        if not found_keywords and len(response.episodic_events) > 0:
            print(f"  Retrieved {len(response.episodic_events)} events, first few:")
            for ev in response.episodic_events[:3]:
                print(f"    - Turn {ev.turn_id}: {ev.content[:80]}...")

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL RECALL TESTS PASSED!")
    else:
        print("SOME RECALL TESTS FAILED")
    print("=" * 60)

    await postgres.close()
    await redis.close()
    await neo4j.close()

    return all_passed


if __name__ == "__main__":
    success = asyncio.run(run_long_conversation_test())
    exit(0 if success else 1)
