"""
Configuration management for XABSA project.
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


class Config:
    """Configuration class for managing experiment configs."""

    def __init__(self, config_path: str):
        """
        Initialize config from YAML file.

        Args:
            config_path: Path to config YAML file
        """
        self.config_path = config_path
        self.config = self.load_config(config_path)

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load config from YAML file.

        Supports inheritance with _base_ key.

        Args:
            config_path: Path to config file

        Returns:
            Config dictionary
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # Handle inheritance
        if '_base_' in config:
            base_path = config.pop('_base_')
            # Resolve relative path
            base_path = Path(config_path).parent / base_path
            base_config = self.load_config(str(base_path))

            # Merge configs (current config overrides base)
            config = self._merge_configs(base_config, config)

        return config

    def _merge_configs(self, base: Dict, override: Dict) -> Dict:
        """
        Merge two config dictionaries.

        Args:
            base: Base config
            override: Override config

        Returns:
            Merged config
        """
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value

        return result

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get config value by key (supports nested keys with dot notation).

        Args:
            key: Config key (e.g., "model.backbone")
            default: Default value if key not found

        Returns:
            Config value
        """
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def __getitem__(self, key: str) -> Any:
        """Get config value by key."""
        return self.config[key]

    def __contains__(self, key: str) -> bool:
        """Check if key exists in config."""
        return key in self.config

    def to_dict(self) -> Dict[str, Any]:
        """Return config as dictionary."""
        return self.config


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load config from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Config dictionary
    """
    config = Config(config_path)
    return config.to_dict()
