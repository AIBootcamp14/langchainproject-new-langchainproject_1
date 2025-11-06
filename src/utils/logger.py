# src/utils/logger.py
"""
Logging Configuration

애플리케이션 전역 로깅 설정을 제공합니다.
파일과 콘솔에 동시에 로그를 출력하며, 레벨별 포맷을 지원합니다.
"""

import logging
import sys
from datetime import datetime
from typing import Optional

from src.utils.config import Config

def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    로거 인스턴스를 생성하여 반환합니다.
    
    Args:
        name: 로거 이름 (일반적으로 __name__ 사용)
        level: 로그 레벨 ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
               None일 경우 .env 파일에 설정된 level 혹은 기본값(INFO) 사용
    
    Returns:
        설정된 Logger 인스턴스
    
    Example:
        >>> from utils.logger import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("This is an info message")
    """
    # 로거 생성
    logger = logging.getLogger(name)
    
    # 이미 핸들러가 설정되어 있으면 재설정 방지
    if logger.handlers:
        return logger
    
    # 로그 레벨 설정
    log_level = getattr(logging, (level or "INFO").upper())
    logger.setLevel(log_level)
    
    # 로그 포맷 설정
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 콘솔 핸들러 (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Config에서 로그 디렉토리 가져오기
    log_dir = Config.LOGS_DIR
    log_dir.mkdir(exist_ok=True, parents=True) 
    
    # 날짜별 로그 파일
    log_file = log_dir / f"app_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 에러 전용 로그 파일
    error_log_file = log_dir / f"error_{datetime.now().strftime('%Y%m%d')}.log"
    error_handler = logging.FileHandler(error_log_file, encoding='utf-8')
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)
    
    # 상위 로거로 전파하지 않음 (중복 로그 방지)
    logger.propagate = False
    
    return logger


def setup_root_logger(level: str = "INFO"):
    """
    루트 로거를 설정합니다.
    애플리케이션 시작 시 한 번만 호출하면 됩니다.
    
    Args:
        level: 로그 레벨 ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
    
    Example:
        >>> from utils.logger import setup_root_logger
        >>> setup_root_logger("DEBUG")
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


# 테스트 코드
if __name__ == "__main__":
    # 테스트
    logger = get_logger(__name__)
    
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
    
    print("\n로그 파일 생성:")
    print(f"- logs/app_{datetime.now().strftime('%Y%m%d')}.log")
    print(f"- logs/error_{datetime.now().strftime('%Y%m%d')}.log")