"""
Main entry point for the Literature Retrieval Engine.

Usage:
    # Build index from papers
    python project/main.py --mode build --papers-dir ./sample_data/papers
    
    # Search for similar papers
    python project/main.py --mode search --query "neural networks"
"""

import argparse
from pathlib import Path
from typing import Optional

from project.retrieval.retriever import PaperRetriever
from project.utils.config import RetrieverConfig
from project.utils.logger import Logger

logger = Logger.get_logger(__name__)


def build_index(
    retriever: PaperRetriever,
    papers_dir: Optional[Path] = None,
    incremental: bool = False
) -> None:
    """Build or update the retrieval index."""
    try:
        if incremental:
            count = retriever.incremental_index(papers_dir)
        else:
            count = retriever.build_retrieval_index(papers_dir)
        
        if count > 0:
            stats = retriever.get_index_stats()
            logger.info(f"✓ Index built successfully: {stats}")
        else:
            logger.error("Failed to build index")
            
    except Exception as e:
        logger.error(f"Error building index: {e}")
        raise


def search_papers(
    retriever: PaperRetriever,
    query: str,
    top_k: int = 5
) -> None:
    """Search for similar papers."""
    try:
        # Load index if not already loaded
        if retriever.index_manager.index is None:
            if not retriever.load_retrieval_index():
                logger.error("Cannot search: no index available")
                return
        
        results = retriever.search_by_text(query, top_k)
        
        if results:
            logger.info(f"\n{'='*80}")
            logger.info(f"Query: {query}")
            logger.info(f"Found {len(results)} similar papers")
            logger.info(f"{'='*80}\n")
            
            for result in results:
                logger.info(f"[{result['rank']}] Score: {result['score']:.4f}")
                logger.info(f"    Title: {result['title']}")
                logger.info(f"    Year:  {result['year']}")
                logger.info(f"    Authors: {', '.join(result['authors'][:3])}")
                logger.info(f"    References: {result['reference_count']}")
                logger.info("")
        else:
            logger.info("No results found")
            
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise


def show_stats(retriever: PaperRetriever) -> None:
    """Display index statistics."""
    try:
        if not retriever.load_retrieval_index():
            logger.warning("No index found")
            return
        
        stats = retriever.get_index_stats()
        logger.info("\n" + "="*80)
        logger.info("INDEX STATISTICS")
        logger.info("="*80)
        
        for key, value in stats.items():
            if key == "config":
                logger.info("\nConfiguration:")
                for cfg_key, cfg_val in value.items():
                    logger.info(f"  {cfg_key}: {cfg_val}")
            else:
                logger.info(f"{key}: {value}")
        
        logger.info("="*80 + "\n")
        
    except Exception as e:
        logger.error(f"Error retrieving stats: {e}")
        raise


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Literature Retrieval Engine for Research Papers"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["build", "search", "stats", "incremental"],
        default="search",
        help="Operation mode"
    )
    
    parser.add_argument(
        "--papers-dir",
        type=Path,
        help="Directory containing paper JSON files"
    )
    
    parser.add_argument(
        "--query",
        type=str,
        help="Search query"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top results"
    )
    
    parser.add_argument(
        "--config-model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Embedding model"
    )
    
    args = parser.parse_args()
    
    # Setup configuration
    config = RetrieverConfig(
        embedding_model=args.config_model,
        papers_dir=args.papers_dir or RetrieverConfig().papers_dir,
        log_file=Path("logs/retriever.log")
    )
    
    # Initialize retriever
    retriever = PaperRetriever(config)
    
    # Execute mode
    if args.mode == "build":
        logger.info("Building index...")
        build_index(retriever, args.papers_dir)
    
    elif args.mode == "incremental":
        logger.info("Incremental indexing...")
        build_index(retriever, args.papers_dir, incremental=True)
    
    elif args.mode == "search":
        if not args.query:
            logger.error("Search mode requires --query argument")
            return
        search_papers(retriever, args.query, args.top_k)
    
    elif args.mode == "stats":
        show_stats(retriever)


if __name__ == "__main__":
    main()
