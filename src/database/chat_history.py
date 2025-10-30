# src/database/chat_history.py

"""
Chat History Database Module

이 모듈은 SQLite 데이터베이스를 사용하여 채팅 기록을 안정적으로 관리합니다.
사용자와 에이전트 간의 모든 상호작용은 session_id를 기준으로 저장되며,
대화의 시작(start)과 끝(end) 시점에 기록이 추가됩니다.
동시성 문제 방지 및 안정적인 리소스 관리를 위한 기능이 포함되어 있습니다.
"""

import sqlite3
import os
from datetime import datetime, timezone
from typing import List, Tuple, Optional
from contextlib import contextmanager

# 로거 설정 (프로젝트의 로거를 가져와서 사용한다고 가정)
from src.utils.logger import get_logger
logger = get_logger(__name__)


class ChatHistoryDB:
    """
    SQLite 기반의 채팅 기록 데이터베이스 관리 클래스.
    안전한 동시성 처리와 컨텍스트 관리를 지원합니다.
    """

    def __init__(self, db_path: str = "database/chat.db"):
        """
        데이터베이스에 연결하고 커서를 생성합니다.
        테이블이 없으면 자동으로 생성합니다.

        Args:
            db_path (str): 데이터베이스 파일의 경로.
        """
        try:
            # 데이터베이스 파일이 위치할 디렉토리 생성
            db_dir = os.path.dirname(db_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir)
            
            # check_same_thread=False: Streamlit과 같은 멀티스레드 환경에서의 오류 방지
            self.conn = sqlite3.connect(db_path, check_same_thread=False)
            logger.info(f"데이터베이스에 성공적으로 연결되었습니다: {db_path}")
        except sqlite3.Error as e:
            logger.error(f"데이터베이스 연결 오류: {e}")
            raise

    @contextmanager
    def _get_cursor(self):
        """
        안전한 데이터베이스 작업을 위한 컨텍스트 매니저.
        작업 후 커밋 또는 롤백을 보장하고 커서를 자동으로 닫습니다.
        """
        cursor = None
        try:
            cursor = self.conn.cursor()
            yield cursor
            self.conn.commit()
        except sqlite3.Error as e:
            logger.error(f"데이터베이스 작업 오류: {e}")
            if self.conn:
                self.conn.rollback()
            raise
        finally:
            if cursor:
                cursor.close()

    def setup_database(self):
        """
        'chat_history' 테이블을 생성합니다.
        - session_id: 각 대화 세션을 식별하는 ID
        - role: 메시지를 보낸 주체 (e.g., 'user', 'assistant')
        - content: 메시지의 내용
        - status: 메시지가 기록된 시점 (e.g., 'start', 'end')
        - quality_score: 답변 품질 평가 점수 (선택 사항)
        - timestamp: 메시지가 기록된 시간 (UTC 기준)
        """
        try:
            with self._get_cursor() as cursor:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS chat_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        status TEXT,
                        quality_score REAL,
                        timestamp DATETIME NOT NULL
                    )
                """)
            logger.info("'chat_history' 테이블이 성공적으로 준비되었습니다.")
        except sqlite3.Error as e:
            logger.error(f"테이블 생성 오류: {e}")
            
    def add_message(self, session_id: str, role: str, content: str, status: str, quality_score: Optional[float] = None):
        """
        새로운 채팅 메시지를 데이터베이스에 추가합니다.
        """
        # 타임스탬프를 UTC 기준으로 기록하여 시간대 문제를 방지합니다.
        utc_timestamp = datetime.now(timezone.utc)
        try:
            with self._get_cursor() as cursor:
                cursor.execute("""
                    INSERT INTO chat_history (session_id, role, content, status, quality_score, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (session_id, role, content, status, quality_score, utc_timestamp))
            logger.info(f"새 메시지 저장 완료 - Session ID: {session_id}, Role: {role}")
        except sqlite3.Error as e:
            logger.error(f"메시지 저장 오류: {e}")

    def get_messages_by_session(self, session_id: str) -> List[Tuple]:
        """
        특정 세션 ID에 해당하는 모든 메시지를 시간순으로 조회합니다.
        """
        try:
            with self._get_cursor() as cursor:
                cursor.execute("""
                    SELECT role, content, status, quality_score, timestamp FROM chat_history
                    WHERE session_id = ?
                    ORDER BY timestamp ASC
                """, (session_id,))
                messages = cursor.fetchall()
                logger.info(f"{session_id} 세션에서 {len(messages)}개의 메시지를 조회했습니다.")
                return messages
        except sqlite3.Error as e:
            logger.error(f"메시지 조회 오류: {e}")
            return []

    def close(self):
        """
        데이터베이스 연결을 안전하게 닫습니다.
        """
        if self.conn:
            self.conn.close()
            logger.info("데이터베이스 연결이 안전하게 종료되었습니다.")

