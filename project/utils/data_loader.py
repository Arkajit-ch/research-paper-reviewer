import json
from pathlib import Path
from typing import Dict, List, Iterator, Optional
from dataclasses import dataclass

from .logger import Logger

logger = Logger.get_logger(__name__)


@dataclass
class PaperData:
    """Data class for parsed paper information."""
    paper_id: str
    title: str
    abstract: str
    year: int
    authors: List[str]
    reference_count: int
    file_path: str
    combined_text: str  # title + abstract
    
    def __post_init__(self):
        """Ensure data types are correct."""
        if not isinstance(self.authors, list):
            self.authors = []


class DataLoader:
    """Utility class for loading and parsing paper JSON files."""
    
    @staticmethod
    def load_paper(file_path: Path) -> Optional[PaperData]:
        """
        Load and parse a single paper JSON file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            PaperData object or None if parsing fails
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract fields with defaults
            metadata = data.get("metadata", {})
            title = metadata.get("title", "Unknown")
            abstract = data.get("abstractText", "")
            year = metadata.get("year", 0)
            authors = metadata.get("authors", [])
            references = data.get("references", [])
            
            # Create combined text for embedding
            combined_text = f"{title}. {abstract}".strip()
            
            paper = PaperData(
                paper_id=file_path.stem,
                title=title,
                abstract=abstract,
                year=year,
                authors=authors if isinstance(authors, list) else [],
                reference_count=len(references),
                file_path=str(file_path),
                combined_text=combined_text
            )
            
            return paper
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to parse paper {file_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error loading {file_path}: {e}")
            return None
    
    @staticmethod
    def load_all_papers(
        papers_dir: Path,
        recursive: bool = True
    ) -> Iterator[PaperData]:
        """
        Recursively load all paper JSON files from a directory.
        
        Args:
            papers_dir: Root directory containing paper JSON files
            recursive: Whether to search subdirectories
            
        Yields:
            PaperData objects
        """
        if not papers_dir.exists():
            logger.error(f"Papers directory does not exist: {papers_dir}")
            return
        
        logger.info(f"Loading papers from {papers_dir}...")
        
        # Find all JSON files
        pattern = "**/*.json" if recursive else "*.json"
        json_files = list(papers_dir.glob(pattern))
        
        logger.info(f"Found {len(json_files)} JSON files")
        
        for i, file_path in enumerate(json_files, 1):
            paper = DataLoader.load_paper(file_path)
            if paper:
                logger.debug(f"Loaded [{i}/{len(json_files)}]: {paper.paper_id}")
                yield paper
            else:
                logger.warning(f"Skipped invalid paper: {file_path}")
    
    @staticmethod
    def validate_paper(paper: PaperData) -> bool:
        """
        Validate paper data.
        
        Args:
            paper: PaperData object to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not paper.title or not paper.combined_text:
            logger.warning(f"Paper {paper.paper_id} has empty title or text")
            return False
        if paper.year < 1950 or paper.year > 2100:
            logger.warning(f"Paper {paper.paper_id} has invalid year: {paper.year}")
            return False
        return True
