"""
Real Integration Test - Uses actual NAGA LLM with real documents in workspace
Tests the complete system: memory-system with real NAGA LLM (not mocked)
"""

import asyncio
import sys

sys.path.insert(0, '/Users/neoris/Documents/projects/own/agentic/memory-system')

import asyncpg
import redis.asyncio as aioredis

# Use standalone naga_llm_service (no agents repo dependency)
from tests.naga_llm_service import naga_llm
from memory_system.src.memory_system.config import settings
from memory_system.src.memory_system.services.workspace_service import SourceWorkspaceService


async def test_real_integration():
    print('='*60)
    print('REAL INTEGRATION TEST - AGENTIC SYSTEM')
    print('='*60)
    print(f'NAGA Model: {naga_llm.model}')
    print(f'Embedding Model: {naga_llm.embedding_model}')

    # Connect to DB
    print('\n[1] Connecting to PostgreSQL...')
    pool = await asyncpg.create_pool(
        host=settings.postgres_host,
        port=settings.postgres_port,
        database=settings.postgres_db,
        user=settings.postgres_user,
        password=settings.postgres_password,
        min_size=2,
        max_size=5
    )
    print('    ✓ Connected')

    redis_client = aioredis.Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        db=settings.redis_db,
        decode_responses=True
    )

    service = SourceWorkspaceService(pool, redis_client, naga_llm)

    # Create workspace
    print('\n[2] Creating workspace...')
    owner_id = 'test-owner-real'
    workspace = await service.create_workspace(owner_id, 'Real Integration Test')
    print(f'    ✓ Workspace: {workspace.workspace_id}')

    # Upload REAL document with REAL information
    print('\n[3] Uploading REAL document about banking...')
    doc_content = b'''# Banking Services Documentation

## Payment Processing
- Domestic payments: 2% commission, minimum $1.00
- International payments: 3% commission, 3-5 business days
- Maximum amount: $1,000,000 per transaction

## Refund Policy
- Standard refunds: 5-10 business days
- Expedited refunds: 2-3 business days (extra fee applies)
- No refunds allowed for transactions with fraud indicators

## Claims Department
- Email: claims@bank.com
- Phone: 1-800-BANK-HELP (24/7)
- In-person: Any branch location
- Response time: 48 hours maximum
'''

    source, is_new = await service.register_source(
        workspace_id=workspace.workspace_id,
        source_type='text',
        title='Banking Services Docs',
        content=doc_content,
        mime_type='text/plain'
    )
    print(f'    ✓ Source: {source.source_id}')

    # Wait for processing
    print('\n[4] Waiting for document indexing...')
    for i in range(15):
        await asyncio.sleep(2)
        status = await service.get_source_status(source.source_id)
        current_status = status.get('status', 'unknown')
        print(f'    Attempt {i+1}: status={current_status}')
        if current_status == 'ready':
            print('    ✓ Document indexed and ready!')
            break
    else:
        print(f'    ⚠️ Status still: {current_status}')

    # Test embeddings with NAGA
    print('\n[5] Testing NAGA embeddings...')
    try:
        emb = await naga_llm.embeddings(['banking commission rate'])
        print(f'    ✓ Embedding dim: {len(emb[0])}')
    except Exception as e:
        print(f'    ✗ Embedding error: {e}')

    # Query with REAL NAGA LLM
    print('\n[6] Testing REAL LLM queries on workspace documents...')
    print('-'*60)

    questions = [
        'What is the commission rate for domestic payments?',
        'How long does a standard refund take?',
        'What is the claims email?',
        'What is the maximum payment amount?',
        'Tell me about expedited refunds'
    ]

    results_summary = []

    for q in questions:
        print(f'\n❓ Question: {q}')
        try:
            result = await service.query(
                workspace_id=workspace.workspace_id,
                query=q,
                mode='source_workspace',
                top_chunks=5
            )

            # Check if response is real or generic
            is_real = len(result.response) > 50 and not result.response.startswith('Based on the documents')
            is_empty = len(result.response) < 10

            print(f'📝 Answer: {result.response[:400]}...' if len(result.response) > 400 else f'📝 Answer: {result.response}')
            print(f'   Documents found: {len(result.documents)}')
            print(f'   Chunks retrieved: {len(result.chunks)}')

            if is_empty:
                print('   ⚠️ Empty response - possible indexing issue')
            elif is_real:
                print('   ✅ Real response based on indexed documents')
            else:
                print('   ⚠️ Generic response - may need review')

            results_summary.append({
                'question': q,
                'success': is_real and len(result.chunks) > 0,
                'docs': len(result.documents),
                'chunks': len(result.chunks)
            })

        except Exception as e:
            print(f'   ✗ Error: {e}')
            import traceback
            traceback.print_exc()
            results_summary.append({
                'question': q,
                'success': False,
                'error': str(e)
            })

    # Summary
    print('\n' + '='*60)
    print('TEST SUMMARY')
    print('='*60)
    passed = sum(1 for r in results_summary if r.get('success'))
    total = len(results_summary)
    print(f'Questions answered: {passed}/{total}')

    for r in results_summary:
        status = '✅' if r.get('success') else '❌'
        print(f'  {status} {r["question"][:50]}...')

    # Cleanup
    print('\n[7] Cleaning up...')
    try:
        await service.delete_source(source.source_id)
        print('    ✓ Source deleted')
    except Exception as e:
        print(f'    ⚠️ Cleanup error: {e}')

    await pool.close()
    await redis_client.aclose()

    print('\n' + '='*60)
    print('REAL INTEGRATION TEST COMPLETE')
    print('='*60)

    return passed == total


if __name__ == '__main__':
    success = asyncio.run(test_real_integration())
    sys.exit(0 if success else 1)