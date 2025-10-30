from langchain.tools import tool
from langchain_tavily import TavilySearch


@tool
def write_file(filename: str, content: str) -> str:
    """
    내용(content)을 지정된 파일 이름(filename)으로 저장합니다. 
    분석 결과나 검색 내용을 저장할 때 사용합니다.
    """
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        return f"✅ 성공: '{filename}' 파일이 성공적으로 저장되었습니다."
    except Exception as e:
        return f"❌ 오류: 파일을 저장하는 중 문제가 발생했습니다. {str(e)}"

