"""
Callback mechanisms for the Deep Research process.
"""

from abc import ABC, abstractmethod
import logging
from ..models import ActivityItem, ActivityStatus, SourceItem


class ResearchCallback(ABC):
    """
    Abstract base class for research callbacks.
    Implement this to receive updates on the research process.
    """

    @abstractmethod
    async def on_activity(self, activity: ActivityItem) -> None:
        """
        Called when a new activity occurs.

        Args:
            activity (ActivityItem): The activity that occurred.
        """
        pass

    @abstractmethod
    async def on_source(self, source: SourceItem) -> None:
        """
        Called when a new source is found.

        Args:
            source (SourceItem): The source that was found.
        """
        pass

    @abstractmethod
    async def on_depth_change(
        self, current: int, maximum: int, completed_steps: int, total_steps: int
    ) -> None:
        """
        Called when the depth changes.

        Args:
            current (int): The current depth.
            maximum (int): The maximum depth.
            completed_steps (int): The number of completed steps.
            total_steps (int): The total number of steps.
        """
        pass

    @abstractmethod
    async def on_progress_init(self, max_depth: int, total_steps: int) -> None:
        """
        Called when the progress is initialized.

        Args:
            max_depth (int): The maximum depth.
            total_steps (int): The total number of steps.
        """
        pass

    @abstractmethod
    async def on_finish(self, content: str) -> None:
        """
        Called when the research is finished.

        Args:
            content (str): The final research content.
        """
        pass


class PrintCallback(ResearchCallback):
    """
    A simple callback that prints updates to the console.
    """

    async def on_activity(self, activity: ActivityItem) -> None:
        """Print activity updates."""
        status_symbol = {
            ActivityStatus.PENDING: "⏳",
            ActivityStatus.COMPLETE: "✅",
            ActivityStatus.ERROR: "❌",
        }[activity.status]

        logging.info(
            f"{status_symbol} {activity.type.value.capitalize()}: {activity.message}"
        )

    async def on_source(self, source: SourceItem) -> None:
        """Print source updates."""
        logging.info(f"📄 Source: {source.title} ({source.url})")

    async def on_depth_change(
        self, current: int, maximum: int, completed_steps: int, total_steps: int
    ) -> None:
        """Print depth change updates."""
        progress = int(completed_steps / total_steps * 100) if total_steps > 0 else 0
        logging.info(f"📊 Depth: {current}/{maximum} - Progress: {progress}%")

    async def on_progress_init(self, max_depth: int, total_steps: int) -> None:
        """Print progress initialization."""
        logging.info(
            f"🔍 Research initialized with max depth {max_depth} and {total_steps} total steps"
        )

    async def on_finish(self, content: str) -> None:
        """Print research completion."""
        logging.info(f"✨ Research complete! Result length: {len(content)} characters")
        logging.info("--- Summary of first 200 characters ---")
        logging.info(content[:200] + "..." if len(content) > 200 else content)
