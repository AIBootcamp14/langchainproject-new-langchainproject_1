# src/main.py
"""
Financial AI Agent - CLI 버전
전체 워크플로우를 대화형 CLI로 실행합니다.

사용법:
    python src/main.py
    또는
    uv run python src/main.py
"""

import uuid
import sys
from typing import Optional, List, Dict, Any

from src.workflow.workflow import build_workflow
from src.database.chat_history import ChatHistoryDB
from src.utils.logger import get_logger
from src.utils.workflow_helpers import (
    convert_messages_to_langchain,
    extract_previous_analysis_data,
    process_chart_paths,
    process_file_paths,
    build_response_metadata,
    get_project_root
)

# 로거 초기화
logger = get_logger(__name__)


class FinancialAgentCLI:
    """Financial AI Agent CLI 인터페이스"""

    def __init__(self):
        """초기화: DB, Workflow 설정"""
        print("\n" + "=" * 80)
        print("💰 Financial AI Agent - CLI 버전")
        print("=" * 80)
        print("\n⏳ 시스템 초기화 중...")

        # DB 초기화
        self.db = ChatHistoryDB()
        self.db.setup_database()
        print("✅ 데이터베이스 초기화 완료")

        # Workflow 초기화
        self.workflow = build_workflow()
        print("✅ Workflow 초기화 완료")

        # 세션 관리
        self.session_id: Optional[str] = None
        self.messages: List[Dict[str, Any]] = []

        print("\n🎉 시스템 준비 완료!\n")

    def display_banner(self):
        """환영 메시지 및 도움말 표시"""
        print("\n" + "=" * 80)
        print("📊 금융 AI 어시스턴트에 오신 것을 환영합니다!")
        print("=" * 80)
        print("\n💡 사용 가능한 명령어:")
        print("   - 질문 입력: 금융 관련 질문을 자유롭게 입력하세요")
        print("   - /new      : 새로운 대화 시작")
        print("   - /history  : 이전 대화 목록 보기")
        print("   - /load <ID>: 특정 세션 불러오기")
        print("   - /help     : 도움말 표시")
        print("   - /exit     : 프로그램 종료")
        print("\n📝 예시 질문:")
        print("   - 삼성전자 주식 분석해줘")
        print("   - 애플과 마이크로소프트를 비교하고 차트 그려줘")
        print("   - SPY ETF 차트 보여줘")
        print("=" * 80 + "\n")

    def show_session_list(self):
        """이전 대화 세션 목록 표시"""
        sessions = self.db.get_all_sessions(limit=10)

        if not sessions:
            print("\n📭 아직 저장된 대화가 없습니다.\n")
            return

        print("\n" + "=" * 80)
        print("📚 최근 대화 목록 (최대 10개)")
        print("=" * 80)

        for idx, session_info in enumerate(sessions, 1):
            session_id = session_info["session_id"]
            preview = session_info["preview"]
            message_count = session_info["message_count"]

            # 현재 활성 세션 표시
            is_current = (session_id == self.session_id)
            status = "▶ [현재]" if is_current else f"   {idx}."
            print(f"{status} {preview[:60]}... ({message_count}개 메시지)")
            print(f"      세션 ID: {session_id[:8]}...")

        print("=" * 80)
        print("\n💡 세션 불러오기: /load <번호> 또는 /load <세션ID>\n")

    def load_session(self, session_id: str):
        """특정 세션의 대화 히스토리 로드"""
        history = self.db.get_history(session_id, limit=20)

        if not history:
            print(f"\n⚠️  세션 ID '{session_id[:8]}...'를 찾을 수 없습니다.\n")
            return False

        self.session_id = session_id
        self.messages = []

        # 역순 정렬 (오래된 것부터)
        for msg in reversed(history):
            self.messages.append({
                "role": msg["role"],
                "content": msg["content"],
                "metadata": msg.get("metadata", {})
            })

        print(f"\n✅ 세션 로드 완료: {len(self.messages)}개 메시지")
        print(f"   세션 ID: {session_id[:8]}...\n")

        # 최근 3개 메시지 미리보기
        if len(self.messages) > 0:
            print("📝 최근 대화:")
            for msg in self.messages[-3:]:
                role_icon = "👤" if msg["role"] == "user" else "🤖"
                content_preview = msg["content"][:80]
                print(f"   {role_icon} {content_preview}...")
            print()

        return True

    def create_new_session(self):
        """새로운 대화 세션 시작"""
        self.session_id = str(uuid.uuid4())
        self.messages = []
        print(f"\n✨ 새로운 대화를 시작합니다!")
        print(f"   세션 ID: {self.session_id[:8]}...\n")

    def display_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """메시지 표시 (색상 포함)"""
        if role == "user":
            print("\n" + "─" * 80)
            print("👤 사용자:")
            print("─" * 80)
            print(content)
        else:
            print("\n" + "─" * 80)
            print("🤖 AI 어시스턴트:")
            print("─" * 80)
            print(content)

            # 메타데이터가 있으면 차트/파일 경로 표시
            if metadata:
                if metadata.get("image_paths"):
                    print("\n📊 생성된 차트:")
                    for img_path in metadata["image_paths"]:
                        print(f"   - {img_path}")

                saved_path = metadata.get("pdf_path") or metadata.get("md_path") or metadata.get("txt_path")
                if saved_path:
                    print(f"\n💾 저장된 파일:")
                    print(f"   - {saved_path}")

        print("─" * 80 + "\n")

    def run_workflow(self, question: str) -> Dict[str, Any]:
        """Workflow 실행"""
        print("\n⏳ 분석 중...\n")

        # 이전 메시지를 LangChain 메시지로 변환 (헬퍼 함수 사용)
        previous_messages = convert_messages_to_langchain(self.messages)

        # 가장 최근 assistant 메시지에서 analysis_data 추출 (헬퍼 함수 사용)
        prev_analysis_data = extract_previous_analysis_data(self.messages)

        # Workflow 실행
        result = self.workflow.run(
            question=question,
            session_id=self.session_id,
            previous_messages=previous_messages,
            previous_analysis_data=prev_analysis_data
        )

        return result

    def process_user_input(self, user_input: str):
        """사용자 입력 처리"""
        user_input = user_input.strip()

        # 명령어 처리
        if user_input.startswith("/"):
            command = user_input.split()[0].lower()

            if command == "/help":
                self.display_banner()
                return

            elif command == "/new":
                self.create_new_session()
                return

            elif command == "/history":
                self.show_session_list()
                return

            elif command == "/load":
                parts = user_input.split(maxsplit=1)
                if len(parts) < 2:
                    print("\n⚠️  사용법: /load <번호> 또는 /load <세션ID>\n")
                    return

                target = parts[1].strip()

                # 번호로 로드
                if target.isdigit():
                    idx = int(target) - 1
                    sessions = self.db.get_all_sessions(limit=10)
                    if 0 <= idx < len(sessions):
                        session_id = sessions[idx]["session_id"]
                        self.load_session(session_id)
                    else:
                        print(f"\n⚠️  유효하지 않은 번호입니다: {target}\n")
                else:
                    # 세션 ID로 로드
                    self.load_session(target)
                return

            elif command == "/exit":
                print("\n👋 프로그램을 종료합니다. 감사합니다!\n")
                sys.exit(0)

            else:
                print(f"\n⚠️  알 수 없는 명령어: {command}")
                print("   /help 명령어로 도움말을 확인하세요.\n")
                return

        # 세션이 없으면 새로 생성
        if not self.session_id:
            self.create_new_session()

        # 사용자 메시지 저장
        self.db.add_message(
            session_id=self.session_id,
            role="user",
            content=user_input
        )

        self.messages.append({
            "role": "user",
            "content": user_input,
            "metadata": {}
        })

        # 사용자 메시지 표시
        self.display_message("user", user_input)

        try:
            # Workflow 실행
            result = self.run_workflow(user_input)

            answer = result.get("answer", "")
            quality_passed = result.get("quality_passed", False)

            # 프로젝트 루트 경로 계산 (헬퍼 함수 사용)
            # src/main.py → ai_agent_project (1단계 상위)
            base_path = get_project_root(__file__, levels_up=1)

            # 차트 경로 처리 (헬퍼 함수 사용)
            image_paths = process_chart_paths(result, base_path)

            # 파일 경로 처리 (헬퍼 함수 사용)
            file_paths = process_file_paths(result, base_path)

            # 메타데이터 구성 (헬퍼 함수 사용)
            metadata = build_response_metadata(result, image_paths, file_paths)

            # AI 응답 저장
            self.db.add_message(
                session_id=self.session_id,
                role="assistant",
                content=answer,
                agent_name="report_generator",
                status="success" if quality_passed else "failed",
                quality_score=result.get("quality_detail", {}).get("score"),
                metadata=metadata
            )

            self.messages.append({
                "role": "assistant",
                "content": answer,
                "metadata": metadata
            })

            # AI 응답 표시
            self.display_message("assistant", answer, metadata)

        except Exception as e:
            error_msg = f"""⚠️ 분석 중 오류가 발생했습니다

죄송합니다. 요청을 처리하는 중 문제가 발생했습니다.

가능한 해결 방법:
- 질문을 다르게 표현해보세요
- 더 구체적인 정보를 포함해주세요 (예: 회사명, 날짜 등)
- 잠시 후 다시 시도해주세요

기술적 오류 정보:
{str(e)}
"""
            logger.error(f"CLI workflow 실행 오류: {e}", exc_info=True)

            # 에러 메시지 저장
            self.db.add_message(
                session_id=self.session_id,
                role="assistant",
                content=error_msg,
                agent_name="system",
                status="error"
            )

            self.messages.append({
                "role": "assistant",
                "content": error_msg,
                "metadata": {}
            })

            # 에러 표시
            self.display_message("assistant", error_msg)

    def run(self):
        """CLI 메인 루프 실행"""
        self.display_banner()

        # 기존 세션 표시
        self.show_session_list()

        # 자동으로 새 세션 시작
        self.create_new_session()

        print("💬 질문을 입력하세요 (/help로 도움말 확인)\n")

        while True:
            try:
                # 사용자 입력 받기
                user_input = input("👤 > ").strip()

                if not user_input:
                    continue

                # 종료 명령어
                if user_input.lower() in ['exit', 'quit', '/exit', '/quit']:
                    print("\n👋 프로그램을 종료합니다. 감사합니다!\n")
                    break

                # 입력 처리
                self.process_user_input(user_input)

            except KeyboardInterrupt:
                print("\n\n👋 프로그램을 종료합니다. 감사합니다!\n")
                break

            except EOFError:
                print("\n\n👋 프로그램을 종료합니다. 감사합니다!\n")
                break


def main():
    """메인 함수"""
    try:
        cli = FinancialAgentCLI()
        cli.run()
    except Exception as e:
        logger.error(f"프로그램 실행 오류: {e}", exc_info=True)
        print(f"\n❌ 프로그램 실행 중 오류가 발생했습니다: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
