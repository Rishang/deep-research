"""
Dump management for saving and loading research results.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ...utils import logger


# Custom YAML representer for multi-line strings using | format
def str_representer(dumper, data):
    """
    Custom YAML representer that uses literal block scalar (|) for multi-line strings.
    This makes the YAML output more readable by preserving newlines.
    """
    if isinstance(data, str):
        # Use literal style (|) for strings with newlines or longer than 80 chars
        if "\n" in data or len(data) > 80:
            return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
        # Use plain style for short strings
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


# Register the custom representer for both Dumper and SafeDumper
yaml.add_representer(str, str_representer)
yaml.SafeDumper.add_representer(str, str_representer)


class DumpManager:
    """
    Manages saving and loading research dumps to/from disk.
    Supports both YAML (default) and JSON formats.
    """

    def __init__(self, dump_dir: str = "./dumps", format: str = "yaml"):
        """
        Initialize the dump manager.

        Args:
            dump_dir: Directory to save dumps to. Defaults to './dumps'.
            format: File format to use ('yaml' or 'json'). Defaults to 'yaml'.
        """
        self.dump_dir = Path(dump_dir)
        self.format = format.lower()

        if self.format not in ["yaml", "json"]:
            raise ValueError(f"Unsupported format: {format}. Use 'yaml' or 'json'.")

        # Create dump directory if it doesn't exist
        self.dump_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        session_id: str,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Save data to a dump file.

        Args:
            session_id: Unique session identifier.
            data: Data to save.
            metadata: Optional metadata to include (topic, timestamp, etc.).

        Returns:
            Path to the saved file.
        """
        try:
            # Prepare dump data
            dump_data = {
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                **(metadata or {}),
                **data,
            }

            # Create filename
            extension = ".yaml" if self.format == "yaml" else ".json"
            filename = f"{session_id}{extension}"
            filepath = self.dump_dir / filename

            # Save to file
            with open(filepath, "w", encoding="utf-8") as f:
                if self.format == "yaml":
                    yaml.dump(
                        dump_data,
                        f,
                        Dumper=yaml.Dumper,
                        default_flow_style=False,
                        allow_unicode=True,
                        sort_keys=False,
                        width=1000,  # Prevent line wrapping for better readability
                    )
                else:
                    json.dump(dump_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Research dump saved to: {filepath}")
            return filepath

        except Exception as e:
            raise IOError(f"Failed to save dump: {str(e)}")

    def load(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Load data from a dump file (auto-detects format).

        Args:
            session_id: Session identifier or filename.

        Returns:
            Loaded data dictionary, or None if not found.
        """
        try:
            # Determine file path
            filepath = self._resolve_filepath(session_id)

            if filepath is None or not filepath.exists():
                logger.warning(f"Dump file not found for session: {session_id}")
                return None

            # Load from file (auto-detect format)
            with open(filepath, "r", encoding="utf-8") as f:
                if filepath.suffix == ".json":
                    dump_data = json.load(f)
                else:
                    dump_data = yaml.safe_load(f)

            logger.info(f"Loaded research dump from: {filepath}")
            if "topic" in dump_data:
                logger.info(f"   Topic: {dump_data['topic']}")
            if "timestamp" in dump_data:
                logger.info(f"   Timestamp: {dump_data['timestamp']}")

            return dump_data

        except Exception as e:
            logger.error(f"Error loading dump: {str(e)}")
            return None

    def _resolve_filepath(self, session_id: str) -> Optional[Path]:
        """
        Resolve the filepath for a session ID.
        Tries YAML first, then JSON.

        Args:
            session_id: Session identifier or filename.

        Returns:
            Path to the file, or None if not found.
        """
        # If it's already a full filename
        if session_id.endswith((".yaml", ".yml", ".json")):
            filepath = self.dump_dir / session_id
            if filepath.exists():
                return filepath
            return None

        # Try different extensions
        for ext in [".yaml", ".yml", ".json"]:
            filepath = self.dump_dir / f"{session_id}{ext}"
            if filepath.exists():
                return filepath

        return None

    def list(self) -> List[str]:
        """
        List all available dump files.

        Returns:
            List of session IDs (filenames without extension).
        """
        try:
            if not self.dump_dir.exists():
                return []

            dumps = []
            # Support both YAML (new) and JSON (legacy) formats
            for pattern in ["*.yaml", "*.yml", "*.json"]:
                for filepath in self.dump_dir.glob(pattern):
                    # Remove extension
                    session_id = filepath.stem
                    if session_id not in dumps:
                        dumps.append(session_id)

            return sorted(dumps, reverse=True)  # Most recent first

        except Exception as e:
            logger.error(f"Error listing dumps: {str(e)}")
            return []

    def delete(self, session_id: str) -> bool:
        """
        Delete a dump file.

        Args:
            session_id: Session identifier or filename.

        Returns:
            True if deleted successfully, False otherwise.
        """
        try:
            filepath = self._resolve_filepath(session_id)

            if filepath is None or not filepath.exists():
                logger.warning(f"Dump file not found: {session_id}")
                return False

            filepath.unlink()
            logger.info(f"Deleted dump: {filepath}")
            return True

        except Exception as e:
            logger.error(f"Error deleting dump: {str(e)}")
            return False

    def get_metadata(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata from a dump file without loading all data.

        Args:
            session_id: Session identifier or filename.

        Returns:
            Metadata dictionary or None if not found.
        """
        dump_data = self.load(session_id)
        if dump_data is None:
            return None

        # Extract metadata fields
        metadata = {
            "session_id": dump_data.get("session_id"),
            "topic": dump_data.get("topic"),
            "timestamp": dump_data.get("timestamp"),
            "success": dump_data.get("success"),
            "error": dump_data.get("error"),
        }

        # Add summary stats if available
        if "data" in dump_data:
            data = dump_data["data"]
            metadata.update(
                {
                    "findings_count": len(data.get("findings", [])),
                    "sources_count": len(data.get("sources", [])),
                    "confirmed_facts_count": len(data.get("confirmed_facts", [])),
                    "contradictions_count": len(data.get("contradictions", [])),
                    "completed_steps": data.get("completed_steps"),
                    "total_steps": data.get("total_steps"),
                }
            )

        return metadata


# Convenience functions for backward compatibility
def save_dump(
    session_id: str,
    data: Dict[str, Any],
    dump_dir: str = "./dumps",
    metadata: Optional[Dict[str, Any]] = None,
    format: str = "yaml",
) -> Path:
    """
    Save data to a dump file.

    Args:
        session_id: Unique session identifier.
        data: Data to save.
        dump_dir: Directory to save to.
        metadata: Optional metadata.
        format: File format ('yaml' or 'json').

    Returns:
        Path to the saved file.
    """
    manager = DumpManager(dump_dir, format)
    return manager.save(session_id, data, metadata)


def load_dump(session_id: str, dump_dir: str = "./dumps") -> Optional[Dict[str, Any]]:
    """
    Load data from a dump file.

    Args:
        session_id: Session identifier or filename.
        dump_dir: Directory to load from.

    Returns:
        Loaded data or None if not found.
    """
    manager = DumpManager(dump_dir)
    return manager.load(session_id)


def list_dumps(dump_dir: str = "./dumps") -> List[str]:
    """
    List all available dump files.

    Args:
        dump_dir: Directory to list from.

    Returns:
        List of session IDs.
    """
    manager = DumpManager(dump_dir)
    return manager.list()


def delete_dump(session_id: str, dump_dir: str = "./dumps") -> bool:
    """
    Delete a dump file.

    Args:
        session_id: Session identifier or filename.
        dump_dir: Directory containing dumps.

    Returns:
        True if deleted successfully.
    """
    manager = DumpManager(dump_dir)
    return manager.delete(session_id)
