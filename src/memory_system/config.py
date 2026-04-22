from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # Database
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "graphrag"
    postgres_user: str = "graphrag"
    postgres_password: str = "graphragpass"

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "test1234"

    # NAGA API
    naga_api_key: str = "ng-4P9TvWqfrr1lORngNt5vMAAuoOGVouPR"
    naga_model: str = "sonar:free"

    # MinIO
    minio_endpoint: str = "localhost:9090"
    minio_access_key: str = "minio"
    minio_secret_key: str = "miniosecret"
    minio_bucket: str = "workspace-sources"
    minio_secure: bool = False

    # Memory System
    default_embedding_dim: int = 1536
    active_context_ttl_hours: int = 24
    working_memory_ttl_hours: int = 6
    retrieval_cache_ttl_hours: int = 1

    # Session
    user_id: str = "default-user"
    agent_id: str = "default-agent"

    @property
    def database_url(self) -> str:
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

    @property
    def redis_url(self) -> str:
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"


settings = Settings()
