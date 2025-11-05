# src/utils/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

class Config:
    """환경 변수 및 설정 관리"""
    
    # API Keys
    UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

    # API Key 검증
    @classmethod
    def validate_api_keys(cls):
        """필수 API 키 존재 여부 확인"""
        if not cls.UPSTAGE_API_KEY:
            raise ValueError("UPSTAGE_API_KEY 설정되지 않았습니다.")
        if not cls.TAVILY_API_KEY:
            raise ValueError("TAVILY_API_KEY가 설정되지 않았습니다.")
            
    BASE_DIR = Path(__file__).resolve().parent.parent.parent

    # Logs
    LOGS_DIR = BASE_DIR / "logs"

    # Database
    DB_DIR = BASE_DIR / "database"
    DB_PATH = str(DB_DIR / "chat.db")

    # Vector DB
    PERSIST_DIR = "data/chroma_store"
    COLLECTION_NAME = "finance_terms"
    EMBEDDING_MODEL = "BAAI/bge-m3"
    PDF_PATH_PATTERN = "data/pdf/*.pdf"
    
    # Chunking
    CHUNK_SIZE_S = 300
    CHUNK_OVERLAP_S = 50
    CHUNK_SIZE_L = 800
    CHUNK_OVERLAP_L = 140
    
    # Retrieval
    TOP_K = 5
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # LLM Models
    LLM_MODEL = "solar-pro2"  # Financial Analyst와 Report Generator가 사용할 기본 모델
    LLM_TEMPERATURE = 0  # 기본 temperature (0 = 결정적)

    # Quality Evaluator
    QUALITY_THRESHOLD = 2  # 품질 평가 통과 최저 점수 (1-5점 중)

    # Retriever
    RETRIEVAL_THRESHOLD = 0.3  # 검색 결과 최소 유사도 점수
    DEFAULT_RETRIEVAL_TOP_K = 3  # 기본 검색 결과 개수

    # Chat Session Limits
    MAX_TURNS_PER_SESSION = 10  # 세션당 최대 대화 턴 수 (user+assistant 쌍 10개)

    # Response Messages
    NOT_FINANCE_RESPONSE = "저는 경제, 금융관련 정보를 통해 전문적으로 사용자의 요청을 도와드리는 AI입니다!\n주식, 환율, 기업 분석 등 금융 관련 질문을 해주시면 답변 도와 드릴게요 😄"
    SESSION_LIMIT_RESPONSE = "💬 대화가 너무 길어졌습니다.\n\n왼쪽 사이드바의 '🆕 새 대화' 버튼을 눌러 새로운 대화를 시작해주세요."



