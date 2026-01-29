"""
Prompt management for Deep Research SDK.
Handles loading and rendering prompts from YAML configuration.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import yaml

from ..utils import logger


class PromptManager:
    """
    Manages prompts for the Deep Research SDK.
    Loads prompts from YAML file and renders them with Jinja2.
    """

    def __init__(self, prompts_file: Optional[str] = None):
        """
        Initialize the prompt manager.

        Args:
            prompts_file: Path to prompts YAML file. If None, uses default.
        """
        if prompts_file is None:
            # Use default prompts file in the package
            package_dir = Path(__file__).parent.parent
            prompts_file = str(package_dir / "prompts.yaml.j2")

        self.prompts_file = prompts_file
        self.prompts: Dict[str, Any] = {}
        self._load_prompts()

    def _load_prompts(self) -> None:
        """Load prompts from YAML file."""
        try:
            with open(self.prompts_file, "r", encoding="utf-8") as f:
                self.prompts = yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.warning(f"Prompts file not found: {self.prompts_file}")
            logger.warning("Using default inline prompts")
            self.prompts = {}
        except Exception as e:
            logger.warning(f"Error loading prompts: {str(e)}")
            logger.warning("Using default inline prompts")
            self.prompts = {}

    def render(self, prompt_key: str, **kwargs) -> str:
        """
        Render a prompt template with the given variables.

        Args:
            prompt_key: Key of the prompt in the YAML file.
            **kwargs: Variables to substitute in the template.

        Returns:
            Rendered prompt string.
        """
        try:
            prompt_config = self.prompts.get(prompt_key, {})
            template = prompt_config.get("template", "")

            if not template:
                raise KeyError(f"Prompt key '{prompt_key}' not found or empty")

            # Simple Jinja2-like variable substitution
            # For more complex needs, you can use actual Jinja2
            rendered = template
            for key, value in kwargs.items():
                placeholder = "{{ " + key + " }}"
                rendered = rendered.replace(placeholder, str(value))

            return rendered

        except Exception as e:
            raise ValueError(f"Error rendering prompt '{prompt_key}': {str(e)}")

    def get_prompt(self, prompt_key: str, **kwargs) -> str:
        """
        Alias for render method for backwards compatibility.

        Args:
            prompt_key: Key of the prompt in the YAML file.
            **kwargs: Variables to substitute in the template.

        Returns:
            Rendered prompt string.
        """
        return self.render(prompt_key, **kwargs)

    def reload(self) -> None:
        """Reload prompts from file. Useful for development/testing."""
        self._load_prompts()

    def list_prompts(self) -> list:
        """
        List all available prompt keys.

        Returns:
            List of prompt keys.
        """
        return list(self.prompts.keys())

    def update_prompt(self, prompt_key: str, template: str) -> None:
        """
        Update a prompt template at runtime.

        Args:
            prompt_key: Key of the prompt to update.
            template: New template string.
        """
        if prompt_key not in self.prompts:
            self.prompts[prompt_key] = {}
        self.prompts[prompt_key]["template"] = template

    def save_prompts(self, filepath: Optional[str] = None) -> None:
        """
        Save current prompts to YAML file.

        Args:
            filepath: Path to save to. If None, uses the loaded file path.
        """
        save_path = filepath or self.prompts_file

        try:
            with open(save_path, "w", encoding="utf-8") as f:
                yaml.dump(self.prompts, f, default_flow_style=False, allow_unicode=True)
            logger.info(f"Prompts saved to: {save_path}")
        except Exception as e:
            logger.error(f"Error saving prompts: {str(e)}")


# Global prompt manager instance
_global_prompt_manager: Optional[PromptManager] = None


def get_prompt_manager(prompts_file: Optional[str] = None) -> PromptManager:
    """
    Get the global prompt manager instance.

    Args:
        prompts_file: Path to prompts file. Only used on first call.

    Returns:
        PromptManager instance.
    """
    global _global_prompt_manager

    if _global_prompt_manager is None:
        _global_prompt_manager = PromptManager(prompts_file)

    return _global_prompt_manager


def render_prompt(prompt_key: str, **kwargs) -> str:
    """
    Convenience function to render a prompt.

    Args:
        prompt_key: Key of the prompt in the YAML file.
        **kwargs: Variables to substitute in the template.

    Returns:
        Rendered prompt string.
    """
    manager = get_prompt_manager()
    return manager.render(prompt_key, **kwargs)
