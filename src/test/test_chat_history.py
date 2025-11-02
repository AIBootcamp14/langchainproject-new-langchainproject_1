# src/test/test_chat_history.py
"""
Chat History 개별 테스트

SQLite 데이터베이스를 사용하여 채팅 기록을 저장하고 조회하는 기능을 테스트합니다.
"""

import os
import tempfile
from src.database.chat_history import ChatHistoryDB
from src.utils.logger import get_logger

logger = get_logger(__name__)


def test_chat_history():
    """Chat History 단독 테스트"""

    logger.info("=" * 50)
    logger.info("Chat History 개별 테스트 시작")
    logger.info("=" * 50)

    # 임시 DB 파일 사용 (테스트 후 자동 삭제)
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
        test_db_path = tmp_file.name

    try:
        # ChatHistoryDB 초기화
        db = ChatHistoryDB(db_path=test_db_path)
        db.setup_database()
        logger.info("ChatHistoryDB 초기화 및 테이블 생성 성공")

        # 테스트용 세션 ID
        session_id = "test-session-123"

        # 테스트 케이스 정의
        test_cases = []

        # [테스트 1] 사용자 메시지 추가
        logger.info("\n[테스트 1] 사용자 메시지 추가")
        db.add_message(
            session_id=session_id,
            role="user",
            content="애플 주식 현재가 알려줘"
        )
        test_cases.append({
            "description": "사용자 메시지 추가",
            "passed": True
        })
        logger.info("✅ PASS - 사용자 메시지 저장 완료")

        # [테스트 2] Assistant 메시지 추가 (성공)
        logger.info("\n[테스트 2] Assistant 메시지 추가 (성공)")
        db.add_message(
            session_id=session_id,
            role="assistant",
            content="Apple Inc. (AAPL)의 현재 주가는 $221.40입니다.",
            agent_name="financial_analyst",
            status="success",
            quality_score=0.95,
            metadata={"ticker": "AAPL", "price": 221.40}
        )
        test_cases.append({
            "description": "Assistant 메시지 추가 (성공)",
            "passed": True
        })
        logger.info("✅ PASS - Assistant 메시지 저장 완료 (성공)")

        # [테스트 3] 사용자 메시지 추가 (두 번째)
        logger.info("\n[테스트 3] 사용자 메시지 추가 (두 번째)")
        db.add_message(
            session_id=session_id,
            role="user",
            content="테슬라 주가는?"
        )
        test_cases.append({
            "description": "사용자 메시지 추가 (두 번째)",
            "passed": True
        })
        logger.info("✅ PASS - 사용자 메시지 저장 완료")

        # [테스트 4] Assistant 메시지 추가 (실패 - empty)
        logger.info("\n[테스트 4] Assistant 메시지 추가 (실패 - empty)")
        db.add_message(
            session_id=session_id,
            role="assistant",
            content="",
            agent_name="financial_analyst",
            status="failed",
            failure_reason="empty",
            quality_score=0.0
        )
        test_cases.append({
            "description": "Assistant 메시지 추가 (실패 - empty)",
            "passed": True
        })
        logger.info("✅ PASS - Assistant 실패 메시지 저장 완료")

        # [테스트 5] 사용자 메시지 추가 (세 번째)
        logger.info("\n[테스트 5] 사용자 메시지 추가 (세 번째)")
        db.add_message(
            session_id=session_id,
            role="user",
            content="마이크로소프트 분석해줘"
        )
        test_cases.append({
            "description": "사용자 메시지 추가 (세 번째)",
            "passed": True
        })
        logger.info("✅ PASS - 사용자 메시지 저장 완료")

        # [테스트 6] Assistant 메시지 추가 (실패 - error)
        logger.info("\n[테스트 6] Assistant 메시지 추가 (실패 - error)")
        db.add_message(
            session_id=session_id,
            role="assistant",
            content="오류가 발생했습니다. 데이터를 가져올 수 없습니다.",
            agent_name="financial_analyst",
            status="failed",
            failure_reason="error",
            quality_score=0.1
        )
        test_cases.append({
            "description": "Assistant 메시지 추가 (실패 - error)",
            "passed": True
        })
        logger.info("✅ PASS - Assistant 실패 메시지 저장 완료")

        # [테스트 7] get_history() - 성공한 메시지만 조회
        logger.info("\n[테스트 7] get_history() - 성공한 메시지만 조회")
        history = db.get_history(session_id, limit=10)

        # 성공한 메시지는 3개여야 함 (user, assistant 성공, user)
        success_messages = [msg for msg in history if msg is not None]
        expected_success_count = 3  # user 3개 + assistant 1개 성공 = 4개인데, DESC 정렬이므로 최근 3개

        # 실제로는 성공한 assistant 메시지가 1개, 성공한 user 메시지가 3개 = 총 4개
        # DESC 정렬이므로 최신 것부터 나옴
        passed = len(success_messages) == 4
        test_cases.append({
            "description": "get_history() - 성공 메시지만 조회",
            "passed": passed,
            "expected": 4,
            "actual": len(success_messages)
        })

        if passed:
            logger.info(f"✅ PASS - 성공 메시지 {len(success_messages)}개 조회 완료")
            logger.info(f"첫 번째 메시지 (최신): {success_messages[0]['content'][:50]}...")
        else:
            logger.error(f"❌ FAIL - 예상: 4개, 실제: {len(success_messages)}개")

        # [테스트 8] get_statistics() - 통계 조회
        logger.info("\n[테스트 8] get_statistics() - 통계 조회")
        stats = db.get_statistics(session_id)

        # 예상: 총 6개 (user 3 + assistant 3), 성공 4개, 실패 2개
        expected_total = 6
        expected_success = 4
        expected_failed = 2

        passed = (
            stats["total_messages"] == expected_total and
            stats["success_count"] == expected_success and
            stats["failed_count"] == expected_failed and
            stats["success_rate"] == (expected_success / expected_total) and
            "empty" in stats["failure_reasons"] and
            "error" in stats["failure_reasons"]
        )

        test_cases.append({
            "description": "get_statistics() - 통계 조회",
            "passed": passed,
            "stats": stats
        })

        if passed:
            logger.info(f"✅ PASS - 통계 조회 성공")
            logger.info(f"  전체: {stats['total_messages']}, 성공: {stats['success_count']}, 실패: {stats['failed_count']}")
            logger.info(f"  성공률: {stats['success_rate']:.1%}")
            logger.info(f"  실패 이유: {stats['failure_reasons']}")
        else:
            logger.error(f"❌ FAIL - 통계가 예상과 다름")
            logger.error(f"  예상: 전체={expected_total}, 성공={expected_success}, 실패={expected_failed}")
            logger.error(f"  실제: {stats}")

        # [테스트 9] clear_session() - 세션 삭제
        logger.info("\n[테스트 9] clear_session() - 세션 삭제")
        db.clear_session(session_id)

        # 삭제 후 조회하면 0개여야 함
        history_after_clear = db.get_history(session_id, limit=10)
        passed = len(history_after_clear) == 0

        test_cases.append({
            "description": "clear_session() - 세션 삭제",
            "passed": passed
        })

        if passed:
            logger.info("✅ PASS - 세션 삭제 완료 (삭제 후 조회 시 0개)")
        else:
            logger.error(f"❌ FAIL - 삭제 후에도 {len(history_after_clear)}개 메시지 존재")

        # DB 연결 종료
        db.close()

        # 결과 요약
        logger.info("\n" + "=" * 50)
        logger.info("테스트 결과 요약")
        logger.info("=" * 50)

        total = len(test_cases)
        passed_count = sum(1 for tc in test_cases if tc["passed"])
        failed_count = total - passed_count

        logger.info(f"전체: {total}개")
        logger.info(f"성공: {passed_count}개")
        logger.info(f"실패: {failed_count}개")
        logger.info(f"성공률: {(passed_count/total*100):.1f}%")

        if failed_count > 0:
            logger.warning("\n실패한 테스트:")
            for tc in test_cases:
                if not tc["passed"]:
                    logger.warning(f"- {tc['description']}")
                    if "expected" in tc and "actual" in tc:
                        logger.warning(f"  예상: {tc['expected']}, 실제: {tc['actual']}")

        return passed_count == total

    finally:
        # 테스트 DB 파일 삭제
        if os.path.exists(test_db_path):
            os.unlink(test_db_path)
            logger.info(f"\n테스트 DB 파일 삭제: {test_db_path}")


if __name__ == "__main__":
    success = test_chat_history()
    exit(0 if success else 1)
