from typing import List

import bs4
import faiss
from indexer.embeddings_models import Gemma300MEmbeddings, OpenAIEmbedding
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from textwrap import shorten
from langchain_core.documents import Document

SAMPLE_DOC_URL = "https://lilianweng.github.io/posts/2023-06-23-agent/"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def load_web_content(web_path: str) -> List[str]:
    """Load web content using a BeautifulSoup strainer to filter specific HTML elements.
      web_path: The URL of the web page to load. https://lilianweng.github.io/posts/2023-06-23-agent/ 
      Returns a list of documents containing the filtered content.
    """
    bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))

    loader = WebBaseLoader(
        web_paths=(web_path,),
        bs_kwargs={"parse_only": bs4_strainer},
    )
    docs = loader.load()
    print("\n[Indexer] Load Web Content")
    print(f"  • Loaded {len(docs)} document(s) from {web_path}")
    for idx, doc in enumerate(docs, start=1):
        preview = shorten(doc.page_content.replace("\n", " "), width=80, placeholder="…")
        print(f"    - Doc {idx}: {doc.metadata.get('source', 'N/A')} ({len(doc.page_content)} chars)")
        print(f"      Preview: {preview}")
    return docs


def document_splitter(documents: List[Document]) -> List[Document]:
    """Split the loaded documents into smaller overlapping chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
        # separators=["\n\n", "\n", " ", ""], # Default separators
    )
    all_splits = text_splitter.split_documents(documents)
    print("\n[Indexer] Split Documents")
    print(
        f"  • Produced {len(all_splits)} chunk(s) "
        f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})"
    )
    return all_splits

def init_vector_store():
    """Initialize the vector store (FAISS) to hold document embeddings."""

    embeddings = OpenAIEmbedding()
    embedding_dim = len(embeddings.embed_query("hello world"))
    index = faiss.IndexFlatL2(embedding_dim)

    vector_store = FAISS(
        embedding_function=embeddings.embed_query,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    print("\n[Indexer] Vector store initialization")
    print("  • Initialized FAISS index with OpenAI embeddings")
    return vector_store

def build_vector_store_from_url(url: str=SAMPLE_DOC_URL)-> FAISS:
    """Load, split, and add documents from a web URL to the vector store."""
    docs = load_web_content(web_path=url)
    all_splits = document_splitter(documents=docs)
    vector_store = init_vector_store()
    _ = vector_store.add_documents(documents=all_splits)
    print("\n[Indexer] Persist to Vector Store")
    print(f"  • Added {len(all_splits)} chunk(s) to the FAISS store.")
    return vector_store
