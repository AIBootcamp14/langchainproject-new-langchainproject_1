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
import json
from datetime import datetime, timezone
from typing import List, Tuple, Optional, Dict
from contextlib import contextmanager

# 로거 및 설정 가져오기
from src.utils.logger import get_logger
from src.utils.config import Config

logger = get_logger(__name__)


class ChatHistoryDB:
    """
    SQLite 기반의 채팅 기록 데이터베이스 관리 클래스.
    안전한 동시성 처리와 컨텍스트 관리를 지원합니다.
    """

    def __init__(self, db_path: str = None):
        """
        데이터베이스에 연결하고 커서를 생성합니다.
        테이블이 없으면 자동으로 생성합니다.

        Args:
            db_path (str): 데이터베이스 파일의 경로. None이면 Config.DB_PATH 사용
        """
        # db_path가 제공되지 않으면 Config에서 가져오기
        if db_path is None:
            db_path = Config.DB_PATH
            logger.info(f"기본 DB 경로 사용: {db_path}")
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
        - id: INTEGER PRIMARY KEY (자동 증가)
        - session_id: TEXT NOT NULL (브라우저 탭별 고유 ID)
        - timestamp: DATETIME DEFAULT CURRENT_TIMESTAMP (저장 시각)
        - role: TEXT NOT NULL (user, assistant, system 중 하나)
        - content: TEXT NOT NULL (메시지 내용)
        - agent_name: TEXT (어느 에이전트가 응답했는지, NULL 가능)
        - status: TEXT DEFAULT 'success' (success 또는 failed)
        - failure_reason: TEXT (실패 시 이유: empty, error, incorrect 중 하나, NULL 가능)
        - quality_score: REAL (품질 점수 0.0~1.0, NULL 가능)
        - metadata: TEXT (추가 정보를 JSON 문자열로 저장, NULL 가능)
        """
        try:
            with self._get_cursor() as cursor:
                # 테이블 생성
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS chat_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        agent_name TEXT,
                        status TEXT DEFAULT 'success',
                        failure_reason TEXT,
                        quality_score REAL,
                        metadata TEXT
                    )
                """)

                # 인덱스 생성
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_session_id
                    ON chat_history(session_id)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_timestamp
                    ON chat_history(timestamp)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_status
                    ON chat_history(status)
                """)

            logger.info("'chat_history' 테이블 및 인덱스가 성공적으로 준비되었습니다.")
        except sqlite3.Error as e:
            logger.error(f"테이블/인덱스 생성 오류: {e}")
            
    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        agent_name: Optional[str] = None,
        status: str = "success",
        failure_reason: Optional[str] = None,
        quality_score: Optional[float] = None,
        metadata: Optional[Dict] = None
    ):
        """
        새로운 채팅 메시지를 데이터베이스에 추가합니다.

        Args:
            session_id: 세션 고유 ID (필수)
            role: "user" 또는 "assistant" (필수)
            content: 메시지 본문 (필수)
            agent_name: 에이전트 이름 (선택)
            status: "success" 또는 "failed" (기본값: "success")
            failure_reason: 실패 이유 - empty/error/incorrect 중 하나 (선택)
            quality_score: 품질 점수 0.0~1.0 (선택)
            metadata: 추가 정보 딕셔너리 (선택, JSON으로 저장됨)
        """
        # 타임스탬프를 UTC 기준으로 기록하여 시간대 문제를 방지합니다.
        utc_timestamp = datetime.now(timezone.utc)

        # metadata를 JSON 문자열로 변환
        metadata_json = json.dumps(metadata) if metadata else None

        try:
            with self._get_cursor() as cursor:
                cursor.execute("""
                    INSERT INTO chat_history
                    (session_id, role, content, agent_name, status, failure_reason, quality_score, metadata, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (session_id, role, content, agent_name, status, failure_reason, quality_score, metadata_json, utc_timestamp))
            logger.info(f"새 메시지 저장 완료 - Session ID: {session_id}, Role: {role}, Status: {status}")
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

    def get_history(self, session_id: str, limit: int = 10) -> List[Dict]:
        """
        컨텍스트로 사용할 최근 대화 가져오기 (성공한 것만).

        Args:
            session_id: 세션 고유 ID
            limit: 최근 N개만 가져오기 (기본값: 10)

        Returns:
            딕셔너리 리스트 (각 딕셔너리는 한 메시지)
            예: [{"role": "user", "content": "...", "timestamp": "...", ...}, ...]
        """
        try:
            with self._get_cursor() as cursor:
                cursor.execute("""
                    SELECT role, content, agent_name, timestamp, metadata
                    FROM chat_history
                    WHERE session_id = ? AND status = 'success'
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (session_id, limit))
                rows = cursor.fetchall()

                # 딕셔너리 리스트로 변환
                messages = []
                for row in rows:
                    role, content, agent_name, timestamp, metadata_json = row
                    message = {
                        "role": role,
                        "content": content,
                        "agent_name": agent_name,
                        "timestamp": timestamp
                    }
                    # metadata가 있으면 JSON 파싱
                    if metadata_json:
                        try:
                            message["metadata"] = json.loads(metadata_json)
                        except json.JSONDecodeError:
                            logger.warning(f"metadata JSON 파싱 실패: {metadata_json}")
                            message["metadata"] = None
                    else:
                        message["metadata"] = None

                    messages.append(message)

                logger.info(f"{session_id} 세션에서 성공한 메시지 {len(messages)}개를 조회했습니다 (최대 {limit}개).")
                return messages
        except sqlite3.Error as e:
            logger.error(f"대화 기록 조회 오류: {e}")
            return []

    def get_statistics(self, session_id: str) -> Dict:
        """
        세션의 통계 정보 제공.

        Args:
            session_id: 세션 고유 ID

        Returns:
            딕셔너리:
            - total_messages: 전체 메시지 수
            - success_count: 성공 메시지 수
            - failed_count: 실패 메시지 수
            - success_rate: 성공률 (0.0~1.0)
            - failure_reasons: 실패 이유별 카운트 딕셔너리
        """
        try:
            with self._get_cursor() as cursor:
                # 전체 통계
                cursor.execute("""
                    SELECT
                        COUNT(*) as total,
                        SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as success,
                        SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed
                    FROM chat_history
                    WHERE session_id = ?
                """, (session_id,))
                row = cursor.fetchone()
                total, success, failed = row if row else (0, 0, 0)

                # 실패 이유별 분포
                cursor.execute("""
                    SELECT failure_reason, COUNT(*) as count
                    FROM chat_history
                    WHERE session_id = ? AND status = 'failed' AND failure_reason IS NOT NULL
                    GROUP BY failure_reason
                """, (session_id,))
                failure_reasons = {reason: count for reason, count in cursor.fetchall()}

                # 성공률 계산
                success_rate = (success / total) if total > 0 else 0.0

                stats = {
                    "total_messages": total,
                    "success_count": success,
                    "failed_count": failed,
                    "success_rate": success_rate,
                    "failure_reasons": failure_reasons
                }

                logger.info(f"{session_id} 세션 통계 조회 완료 - 전체: {total}, 성공: {success}, 실패: {failed}")
                return stats
        except sqlite3.Error as e:
            logger.error(f"통계 조회 오류: {e}")
            return {
                "total_messages": 0,
                "success_count": 0,
                "failed_count": 0,
                "success_rate": 0.0,
                "failure_reasons": {}
            }

    def get_all_sessions(self, limit: int = 10) -> List[Dict]:
        """
        모든 세션 목록을 최근 순으로 가져옵니다.

        Args:
            limit: 최대 세션 개수 (기본값: 10)

        Returns:
            세션 정보 딕셔너리 리스트
            각 딕셔너리: {
                "session_id": str,
                "preview": str (첫 번째 user 메시지 미리보기),
                "message_count": int,
                "last_timestamp": str
            }
        """
        try:
            with self._get_cursor() as cursor:
                # 각 세션의 첫 번째 user 메시지와 메시지 수 가져오기
                cursor.execute("""
                    SELECT
                        session_id,
                        COUNT(*) as message_count,
                        MAX(timestamp) as last_timestamp,
                        (SELECT content FROM chat_history c2
                         WHERE c2.session_id = c1.session_id AND c2.role = 'user'
                         ORDER BY c2.timestamp ASC LIMIT 1) as first_content
                    FROM chat_history c1
                    GROUP BY session_id
                    ORDER BY last_timestamp DESC
                    LIMIT ?
                """, (limit,))

                rows = cursor.fetchall()

                sessions = []
                for row in rows:
                    session_id, message_count, last_timestamp, first_content = row
                    # 미리보기 텍스트 자르기 (최대 50자)
                    preview = first_content if first_content else "(빈 대화)"
                    preview_short = preview[:50] + "..." if len(preview) > 50 else preview

                    sessions.append({
                        "session_id": session_id,
                        "preview": preview_short,
                        "message_count": message_count,
                        "last_timestamp": last_timestamp
                    })

                logger.info(f"{len(sessions)}개의 세션을 조회했습니다.")
                return sessions
        except sqlite3.Error as e:
            logger.error(f"세션 목록 조회 오류: {e}")
            return []

    def get_turn_count(self, session_id: str) -> int:
        """
        세션의 대화 턴 수를 반환합니다.
        턴 수는 user 메시지 개수로 계산합니다.

        Args:
            session_id: 세션 고유 ID

        Returns:
            int: 대화 턴 수 (user 메시지 개수)
        """
        try:
            with self._get_cursor() as cursor:
                cursor.execute("""
                    SELECT COUNT(*) FROM chat_history
                    WHERE session_id = ? AND role = 'user'
                """, (session_id,))
                count = cursor.fetchone()[0]
                logger.debug(f"{session_id} 세션의 턴 수: {count}")
                return count
        except sqlite3.Error as e:
            logger.error(f"턴 수 조회 오류: {e}")
            return 0

    def clear_session(self, session_id: str):
        """
        특정 세션의 기록 삭제.

        Args:
            session_id: 세션 고유 ID
        """
        try:
            with self._get_cursor() as cursor:
                cursor.execute("""
                    DELETE FROM chat_history
                    WHERE session_id = ?
                """, (session_id,))
                deleted_count = cursor.rowcount
                logger.info(f"{session_id} 세션의 {deleted_count}개 메시지를 삭제했습니다.")
        except sqlite3.Error as e:
            logger.error(f"세션 삭제 오류: {e}")

    def close(self):
        """
        데이터베이스 연결을 안전하게 닫습니다.
        """
        if self.conn:
            self.conn.close()
            logger.info("데이터베이스 연결이 안전하게 종료되었습니다.")

