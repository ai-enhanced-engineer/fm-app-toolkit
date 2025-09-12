"""Example demonstrating LocalDocumentRepository usage with document loading and chunking."""

import argparse
from pathlib import Path

from llama_index.core.node_parser import SentenceSplitter

from src.logging import get_logger

from .local import LocalDocumentRepository

logger = get_logger(__name__)


def process_documents(data_path: str) -> None:
    """Load documents from directory and demonstrate chunking with structured logging."""
    test_data_path = Path(data_path)

    # Opening print statement
    print("\n📚 LocalDocumentRepository Example")
    print("=" * 50)
    print("This example demonstrates:")
    print("  • Loading documents from local filesystem")
    print("  • Processing with structured logging")
    print("  • Chunking text with SentenceSplitter")
    print(f"  • Data path: {test_data_path}")
    print("=" * 50 + "\n")

    # Start with structured logging
    logger.info("Starting LocalDocumentRepository demonstration")

    # Initialize repository

    repo = LocalDocumentRepository(
        input_dir=str(test_data_path),
        recursive=True,
        required_exts=[".txt"],
    )
    logger.info("Initialized LocalDocumentRepository", path=str(test_data_path))

    # Load documents
    logger.info("Starting document loading", repository_type="LocalDocumentRepository")

    documents = repo.load_documents(location=str(test_data_path))

    logger.info(
        "Documents loaded successfully",
        document_count=len(documents),
        total_chars=sum(len(doc.text) for doc in documents),
    )

    # Log document details
    for i, doc in enumerate(documents):
        logger.debug(
            "Document details",
            doc_index=i,
            char_count=len(doc.text),
            metadata_keys=list(doc.metadata.keys()) if doc.metadata else [],
        )

    # Perform chunking
    logger.info("Starting text chunking", chunk_size=512, chunk_overlap=50)

    text_splitter = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=50,
    )

    nodes = text_splitter.get_nodes_from_documents(documents)

    logger.info("Text chunking completed", original_docs=len(documents), total_chunks=len(nodes))

    # Log sample chunk details
    if nodes:
        sample_chunk = nodes[0]
        logger.debug(
            "Sample chunk details",
            chunk_length=len(sample_chunk.get_content()),
            has_metadata=bool(sample_chunk.metadata),
            preview=sample_chunk.get_content()[:100],
        )

    # End with structured logging
    logger.info(
        "LocalDocumentRepository demonstration completed successfully",
        total_documents=len(documents),
        total_chunks=len(nodes),
    )

    # Closing print statement
    print("\n" + "=" * 50)
    print("✅ Example completed successfully!")
    print(f"   • Loaded {len(documents)} documents")
    print(f"   • Created {len(nodes)} chunks")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process documents with loading and chunking demonstration")
    parser.add_argument("--data-path", type=str, help="Path to directory containing documents to load", required=True)
    args = parser.parse_args()

    process_documents(data_path=args.data_path)
