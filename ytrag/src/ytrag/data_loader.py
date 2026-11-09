from pathlib import Path
from typing import Any, List

from langchain_community.document_loaders import CSVLoader, PyPDFLoader, TextLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders import JSONLoader


def load_all_documents(data_dir: str) -> List[Any]:
    """
    Load all supported files from the data directory and convert to LangChain document structure.
    Supported: PDF, TXT, CSV, Excel, Word, JSON
    """
    data_path = Path(data_dir).resolve()
    print(f"[DEBUG] Data path: {data_path}")
    documents = []

    loaders = {
        "pdf": lambda p: PyPDFLoader(p),
        "txt": lambda p: TextLoader(p, encoding="utf-8"),
        "csv": lambda p: CSVLoader(p),
        "xlsx": lambda p: UnstructuredExcelLoader(p),
        "docx": lambda p: Docx2txtLoader(p),
        "json": lambda p: JSONLoader(p, jq_schema=".", text_content=False),
    }

    for ext, make_loader in loaders.items():
        files = list(data_path.glob(f"**/*.{ext}"))
        print(f"[DEBUG] Found {len(files)} {ext.upper()} files: {[str(f) for f in files]}")
        for file in files:
            print(f"[DEBUG] Loading {ext.upper()}: {file}")
            try:
                loaded = make_loader(str(file)).load()
                print(f"[DEBUG] Loaded {len(loaded)} docs from {file}")
                documents.extend(loaded)
            except Exception as e:
                print(f"[ERROR] Failed to load {ext.upper()} {file}: {e}")

    print(f"[DEBUG] Total documents loaded: {len(documents)}")
    return documents