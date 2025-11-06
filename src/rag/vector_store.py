from glob import glob
from pathlib import Path
from typing import List, Optional, Tuple

import statistics as stats

import chromadb
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.config import Config
from src.utils.logger import get_logger

load_dotenv()

logger = get_logger(__name__)

PDF_PATH_PATTERN = Config.PDF_PATH_PATTERN
PERSIST_DIR = Config.PERSIST_DIR
EMBEDDING_MODEL = Config.EMBEDDING_MODEL
COLLECTION_NAME = Config.COLLECTION_NAME
# CHUNK_SIZE_S = Config.CHUNK_SIZE_S
# CHUNK_OVERLAP_S = Config.CHUNK_OVERLAP_S
# CHUNK_SIZE_L = Config.CHUNK_SIZE_L
# CHUNK_OVERLAP_L = Config.CHUNK_OVERLAP_L


class VectorStore:
    def __init__(
        self,
        pdf_path_pattern: str = PDF_PATH_PATTERN,
        persist_dir: str = PERSIST_DIR,
        embedding_model: str = EMBEDDING_MODEL,
        collection_name: str = COLLECTION_NAME,
    ):
        self.pdf_path_pattern = pdf_path_pattern
        self.persist_dir = persist_dir
        self.embedding_model = embedding_model
        self.collection_name = collection_name

        self._embeddings: HuggingFaceEmbeddings | None = None
        self._store: Chroma | None = None

   
    def pick_pdf(self, pdf_path: str | None = None) -> list[str]:
        if pdf_path:
            file_path = Path(pdf_path)
            if not file_path.exists():
                raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {file_path}")
            return [str(file_path)]

        matches = sorted(glob(self.pdf_path_pattern))
        if not matches:
            raise FileNotFoundError(
                f"패턴에 맞는 PDF가 없습니다: {self.pdf_path_pattern}"
            )
        return matches
    
    
    def _decide_chunk_params(med: int):
        # 1) 기본 구간 규칙
        if med < 400:
            chunk_size = 250
            overlap = 30
        elif med < 800:
            chunk_size = 600
            overlap = 90
        elif med < 1500:
            chunk_size = 800
            overlap = 120
        elif med < 3000:
            chunk_size = 1100
            overlap = 180
        else:
            chunk_size = 1400
            overlap = 220

        # 2) med 대비 0.5~0.9배 범위로 미세 조정 (선택)
        #    너무 크거나 작지 않게 클램프
        lo = max(220, int(med * 0.5))
        hi = min(1600, int(med * 0.9))
        chunk_size = max(lo, min(chunk_size, hi))

        # 3) 겹침은 chunk의 ~15%로 맞추되 최소/최대 가이드
        target_overlap = max(25, min(int(chunk_size * 0.15), 260))
        # 위에서 구한 overlap과 타겟을 평균 내 유연화
        overlap = int((overlap + target_overlap) / 2)

        return chunk_size, overlap

    @staticmethod
    def pick_splitter(pages):
        lens = [len(d.page_content or "") for d in pages]
        med = stats.median(lens)
        logger.debug(f"페이지 중간 값: {med}")

        chunk_size, chunk_overlap = VectorStore._decide_chunk_params(med)

        # 용어집/정의집(짧은 항목 위주)일 가능성이 높을 때 분리자 우선
        if med < 800:
            seps = ["\n\n", "\n", " : ", " - ", "—", " ", ""]
        else:
            seps = ["\n\n", "\n", "。", ".", "!", "?", ",", " ", ""]

        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=seps,
        )

    def _ensure_embeddings(self):
        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                encode_kwargs={"normalize_embeddings": True},
            )
        return self._embeddings

    def _ensure_store(self):
        if self._store is None:
            self._store = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self._ensure_embeddings(),
                collection_name=self.collection_name,
            )
        return self._store

    
    def build_vector_store(self, pdf_path: str | None = None):
        pdf_files = self.pick_pdf(pdf_path)
        vectorstore = self._ensure_store()

        all_docs = []
        for pdf_file in pdf_files:
            logger.info(f"PDF 처리 중: {pdf_file}")
            loader = PyPDFLoader(pdf_file)
            pages = list(loader.lazy_load())
            splitter = self.pick_splitter(pages)
            logger.debug(f"Splitter 선택: {splitter}")
            docs = splitter.split_documents(pages)

            for doc in docs:
                doc.metadata["source"] = Path(pdf_file).name
            all_docs.extend(docs)

        if all_docs:
            vectorstore.add_documents(all_docs)
        logger.info(f"총 {len(all_docs)}개 청크 저장 완료 ({len(pdf_files)}개 PDF)")
        return vectorstore

    def similarity_search(self, query: str, top_k: int = 3):
        store = self._ensure_store()
        return store.similarity_search(query, k=top_k)

    def retrieve_with_scores(
        self, query: str, k: Optional[int] = None
    ) -> List[Tuple[Document, float]]:
        store = self._ensure_store()
        top_k = k if k is not None else 3
        return store.similarity_search_with_relevance_scores(query, k=top_k)
    

    def inspect_collections(self):
        client = chromadb.PersistentClient(path=self.persist_dir)
        collections = [c.name for c in client.list_collections()]
        logger.info(f"Collections: {collections}")

        if self.collection_name not in collections:
            logger.warning(f"'{self.collection_name}' 컬렉션을 찾지 못했습니다.")
            return

        collection = client.get_collection(self.collection_name)
        count = collection.count()
        logger.info(f"'{self.collection_name}' 내 문서 수: {count}")

        items = collection.peek()
        logger.info(f"샘플 개수: {len(items['ids'])}")
        for i in range(min(3, len(items["ids"]))):
            logger.debug(f"ID: {items['ids'][i]}")
            logger.debug(f"Metadata: {items['metadatas'][i]}")
            logger.debug(f"Text snippet: {items['documents'][i][:150]}...")


def main():
    store = VectorStore()
    store.build_vector_store()
    store.inspect_collections()


if __name__ == "__main__":
    main()