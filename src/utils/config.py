# src/utils/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

class Config:
    """환경 변수 및 설정 관리"""
    
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    SOLAR_API_KEY = os.getenv("SOLAR_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    
    # API Key 검증
    @classmethod
    def validate_api_keys(cls):
        """필수 API 키 존재 여부 확인"""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
        if not cls.SOLAR_API_KEY:
            raise ValueError("SOLAR_API_KEY가 설정되지 않았습니다.")
        if not cls.TAVILY_API_KEY:
            raise ValueError("TAVILY_API_KEY가 설정되지 않았습니다.")
            
    # Paths
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    VECTOR_DB_PATH = BASE_DIR / "chroma_db"
    LOGS_DIR = BASE_DIR / "logs"
    
    # Embedding
    EMBEDDING_MODEL = "text-embedding-ada-002"
    
    # Chunking
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Retrieval
    TOP_K = 5
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")