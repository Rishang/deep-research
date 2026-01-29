"""
Utility functions for Deep Research SDK.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import logging
import yaml
from rich.logging import RichHandler


def _logger(flag: str = "", format: str = ""):
    if format == "" or format is None:
        format = "%(levelname)s|%(name)s| %(message)s"

    # message
    logger = logging.getLogger(__name__)

    if os.environ.get(flag) is not None:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    # create console handler and set level to debug
    # ch = logging.StreamHandler()
    # ch.setLevel(logging.DEBUG)
    # # create formatter
    # # add formatter to ch
    # formatter = logging.Formatter(format)
    # ch.setFormatter(formatter)

    # # add ch to logger
    # logger.addHandler(ch)
    handler = RichHandler(log_time_format="")
    logger.addHandler(handler)
    return logger


# message
# export LOG_LEVEL=true
logger = _logger("LOG_LEVEL")


def load_prompts_yaml(prompts_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Load prompts from YAML file.

    Args:
        prompts_file: Path to prompts YAML file. If None, uses default.

    Returns:
        Dictionary containing all prompts.
    """
    if prompts_file is None:
        # Use default prompts file in the package
        package_dir = Path(__file__).parent
        prompts_file = str(package_dir / "prompts.j2.yaml")

    try:
        with open(prompts_file, "r", encoding="utf-8") as f:
            prompts = yaml.safe_load(f) or {}
        return prompts
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompts file not found: {prompts_file}")
    except Exception as e:
        raise ValueError(f"Error loading prompts file: {str(e)}")


def render_jinja_prompt(template: str, **kwargs) -> str:
    """
    Render a Jinja2 template with the given variables.
    This is a simple implementation using string replacement.
    For advanced Jinja2 features, install jinja2 package.

    Args:
        template: Template string with {{ variable }} placeholders.
        **kwargs: Variables to substitute in the template.

    Returns:
        Rendered string.
    """
    try:
        # Try to use Jinja2 if available
        from jinja2 import Template as Jinja2Template

        jinja_template = Jinja2Template(template)
        return jinja_template.render(**kwargs)
    except ImportError:
        # Fallback to simple string replacement
        rendered = template
        for key, value in kwargs.items():
            # Handle both {{ var }} and {{var}} formats
            placeholder_with_spaces = "{{ " + key + " }}"
            placeholder_no_spaces = "{{" + key + "}}"
            rendered = rendered.replace(placeholder_with_spaces, str(value))
            rendered = rendered.replace(placeholder_no_spaces, str(value))
        return rendered


def get_prompt_template(prompt_key: str, prompts_file: Optional[str] = None) -> str:
    """
    Get a prompt template by key from the YAML file.

    Args:
        prompt_key: Key of the prompt (e.g., 'exploratory_queries').
        prompts_file: Path to prompts YAML file. If None, uses default.

    Returns:
        Template string.

    Raises:
        KeyError: If prompt key not found.
    """
    prompts = load_prompts_yaml(prompts_file)

    if prompt_key not in prompts:
        raise KeyError(f"Prompt key '{prompt_key}' not found in prompts file")

    prompt_config = prompts[prompt_key]

    if isinstance(prompt_config, dict) and "template" in prompt_config:
        return prompt_config["template"]
    elif isinstance(prompt_config, str):
        return prompt_config
    else:
        raise ValueError(f"Invalid prompt format for key '{prompt_key}'")


def render_prompt(prompt_key: str, prompts_file: Optional[str] = None, **kwargs) -> str:
    """
    Load and render a prompt template with the given variables.

    Args:
        prompt_key: Key of the prompt (e.g., 'exploratory_queries').
        prompts_file: Path to prompts YAML file. If None, uses default.
        **kwargs: Variables to substitute in the template.

    Returns:
        Rendered prompt string.

    Example:
        >>> prompt = render_prompt(
        ...     'exploratory_queries',
        ...     topic='quantum computing'
        ... )
    """
    template = get_prompt_template(prompt_key, prompts_file)
    return render_jinja_prompt(template, **kwargs)


class PromptLoader:
    """
    Prompt loader class for managing and rendering prompts.
    Caches loaded prompts for better performance.
    """

    def __init__(self, prompts_file: Optional[str] = None):
        """
        Initialize the prompt loader.

        Args:
            prompts_file: Path to prompts YAML file. If None, uses default.
        """
        self.prompts_file = prompts_file
        self._prompts_cache: Optional[Dict[str, Any]] = None

    def _load_prompts(self) -> Dict[str, Any]:
        """Load and cache prompts from YAML file."""
        if self._prompts_cache is None:
            self._prompts_cache = load_prompts_yaml(self.prompts_file)
        return self._prompts_cache

    def reload(self) -> None:
        """Reload prompts from file (clears cache)."""
        self._prompts_cache = None
        self._load_prompts()

    def get_template(self, prompt_key: str) -> str:
        """
        Get a prompt template by key.

        Args:
            prompt_key: Key of the prompt.

        Returns:
            Template string.
        """
        prompts = self._load_prompts()

        if prompt_key not in prompts:
            raise KeyError(f"Prompt key '{prompt_key}' not found")

        prompt_config = prompts[prompt_key]

        if isinstance(prompt_config, dict) and "template" in prompt_config:
            return prompt_config["template"]
        elif isinstance(prompt_config, str):
            return prompt_config
        else:
            raise ValueError(f"Invalid prompt format for key '{prompt_key}'")

    def render(self, prompt_key: str, **kwargs) -> str:
        """
        Load and render a prompt template.

        Args:
            prompt_key: Key of the prompt.
            **kwargs: Variables to substitute in the template.

        Returns:
            Rendered prompt string.
        """
        template = self.get_template(prompt_key)
        return render_jinja_prompt(template, **kwargs)

    def list_prompts(self) -> list:
        """
        List all available prompt keys.

        Returns:
            List of prompt keys.
        """
        prompts = self._load_prompts()
        return list(prompts.keys())

    def get_prompt_info(self, prompt_key: str) -> Dict[str, Any]:
        """
        Get full prompt configuration including metadata.

        Args:
            prompt_key: Key of the prompt.

        Returns:
            Prompt configuration dictionary.
        """
        prompts = self._load_prompts()

        if prompt_key not in prompts:
            raise KeyError(f"Prompt key '{prompt_key}' not found")

        return prompts[prompt_key]


# Global prompt loader instance
_global_loader: Optional[PromptLoader] = None


def get_prompt_loader(prompts_file: Optional[str] = None) -> PromptLoader:
    """
    Get the global prompt loader instance.

    Args:
        prompts_file: Path to prompts file. Only used on first call.

    Returns:
        PromptLoader instance.
    """
    global _global_loader

    if _global_loader is None:
        _global_loader = PromptLoader(prompts_file)

    return _global_loader
