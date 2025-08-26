"""PropertyGraph index implementation."""

from typing import Optional

from llama_index.core import Document, PropertyGraphIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.indices.property_graph.transformations import (
    ImplicitPathExtractor,
    SimpleLLMPathExtractor,
)
from llama_index.core.llms import LLM
from llama_index.core.schema import TransformComponent
from pydantic import validate_call

from fm_app_toolkit.logging import get_logger

from .base import DocumentIndexer

logger = get_logger(__name__)


def _select_extractors(
    kg_extractors: Optional[list[TransformComponent]],
    llm: Optional[LLM]
) -> list[TransformComponent]:
    """Choose extractors: LLM-based if available, otherwise implicit only."""
    if kg_extractors is not None:
        return kg_extractors
    
    if llm is not None:
        return [
            SimpleLLMPathExtractor(llm=llm),
            ImplicitPathExtractor(),
        ]
    
    return [ImplicitPathExtractor()]


class PropertyGraphIndexer(DocumentIndexer):
    """Index documents as knowledge graphs for relationship queries."""

    def __init__(
        self,
        kg_extractors: Optional[list[TransformComponent]] = None,
        llm: Optional[LLM] = None,
        embed_kg_nodes: bool = True,
        show_progress: bool = False,
    ):
        self.kg_extractors = kg_extractors
        self.llm = llm
        self.embed_kg_nodes = embed_kg_nodes
        self.show_progress = show_progress

        logger.info(
            "Initializing PropertyGraphIndexer",
            embed_kg_nodes=embed_kg_nodes,
            show_progress=show_progress,
            has_custom_extractors=kg_extractors is not None,
        )

    @validate_call
    def create_index(
        self,
        documents: list[Document],
        embed_model: Optional[BaseEmbedding] = None,
    ) -> PropertyGraphIndex:
        """Build knowledge graph index from documents."""
        try:
            logger.info(f"Creating property graph index from {len(documents)} documents")
            
            # Select appropriate extractors
            kg_extractors = _select_extractors(self.kg_extractors, self.llm)
            
            # Log which extractors are being used for debugging
            logger.debug(
                f"Using extractors: {[type(e).__name__ for e in kg_extractors]}"
            )
            
            # Create index with optional embedding model
            index = PropertyGraphIndex.from_documents(
                documents,
                llm=self.llm,
                kg_extractors=kg_extractors,
                embed_model=embed_model,
                embed_kg_nodes=self.embed_kg_nodes,
                show_progress=self.show_progress,
            )
            
            logger.info(f"Successfully created property graph index with {len(documents)} documents")
            return index
            
        except Exception as e:
            logger.error(f"Failed to create property graph index: {e}")
            raise