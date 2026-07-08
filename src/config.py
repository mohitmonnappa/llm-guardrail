"""
Configuration Module
====================
Loads and manages configuration for LLM guardrails system.
"""

import yaml
import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class Config:
    """Configuration manager for LLM guardrails."""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to config.yaml file
        """
        self.config_path = config_path
        self._config = {}
        self.load()

    def load(self) -> None:
        """Load configuration from YAML file."""
        if not os.path.exists(self.config_path):
            logger.warning(f"Config file not found: {self.config_path}")
            self._config = self._get_defaults()
            return

        try:
            with open(self.config_path, 'r') as f:
                self._config = yaml.safe_load(f) or {}
            logger.info(f"Configuration loaded from {self.config_path}")
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            self._config = self._get_defaults()

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-notation key.

        Examples:
            config.get("model.name")
            config.get("threshold.default")
        """
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default

        return value if value is not None else default

    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by dot-notation key.

        Examples:
            config.set("threshold.default", 0.7)
        """
        keys = key.split(".")
        config = self._config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value
        logger.info(f"Config updated: {key} = {value}")

    def save(self, path: Optional[str] = None) -> None:
        """Save configuration to YAML file."""
        save_path = path or self.config_path

        try:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            with open(save_path, 'w') as f:
                yaml.dump(self._config, f, default_flow_style=False)
            logger.info(f"Configuration saved to {save_path}")
        except Exception as e:
            logger.error(f"Error saving config: {e}")

    def get_model_config(self) -> Dict:
        """Get model configuration section."""
        return self.get("model", {})

    def get_threshold(self) -> float:
        """Get default threshold."""
        return self.get("threshold.default", 0.5)

    def set_threshold(self, threshold: float) -> None:
        """Set default threshold."""
        self.set("threshold.default", threshold)

    def get_categories(self) -> list:
        """Get classification categories."""
        return self.get("categories", [
            "safe",
            "jailbreak",
            "pii_extraction",
            "toxic_content",
            "prompt_injection"
        ])

    def get_paths(self) -> Dict:
        """Get file paths."""
        return self.get("paths", {})

    def get_evaluation_config(self) -> Dict:
        """Get evaluation settings."""
        return self.get("evaluation", {})

    def get_ui_config(self) -> Dict:
        """Get UI settings."""
        return self.get("ui", {})

    def get_training_config(self) -> Dict:
        """Get training settings."""
        return self.get("training", {})

    def _get_defaults(self) -> Dict:
        """Get default configuration."""
        return {
            "model": {
                "name": "distilbert-base-uncased",
                "framework": "transformers",
                "device": "cpu",
                "batch_size": 32,
                "max_length": 512
            },
            "threshold": {
                "default": 0.5
            },
            "categories": [
                "safe",
                "jailbreak",
                "pii_extraction",
                "toxic_content",
                "prompt_injection"
            ],
            "paths": {
                "models_dir": "./models",
                "data_dir": "./data",
                "output_dir": "./output",
                "test_suite": "./data/test_suite.csv",
                "trained_model": "./models/fine_tuned_model"
            },
            "evaluation": {
                "test_set_size": 45,
                "threshold_range": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                "metrics": ["accuracy", "precision", "recall", "f1_score", "false_positive_rate"]
            },
            "ui": {
                "title": "LLM Guardrails Firewall",
                "description": "Classify prompts as safe or unsafe",
                "theme": "light",
                "max_input_length": 2000,
                "show_confidence": True
            },
            "logging": {
                "level": "INFO",
                "log_file": "./output/guardrails.log",
                "log_to_console": True
            },
            "training": {
                "epochs": 3,
                "learning_rate": 2e-5,
                "warmup_steps": 500,
                "save_checkpoint": True
            }
        }

    def to_dict(self) -> Dict:
        """Get entire configuration as dictionary."""
        return self._config

    def __repr__(self) -> str:
        return f"Config(path={self.config_path})"


# Global config instance
_config_instance: Optional[Config] = None


def get_config(config_path: str = "config.yaml") -> Config:
    """Get or create global config instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(config_path)
    return _config_instance


if __name__ == "__main__":
    # Test config
    print("Loading configuration...")
    config = Config("config.yaml")

    print("\nModel config:")
    print(config.get_model_config())

    print("\nThreshold:", config.get_threshold())
    print("Categories:", config.get_categories())

    print("\nAll paths:")
    for key, value in config.get_paths().items():
        print(f"  {key}: {value}")
