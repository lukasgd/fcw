"""Configuration loading and validation."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import yaml


class DirectoryType(str, Enum):
    """Directory type for data flow enforcement.
    
    Values:
        IN: Upload only (input data that flows into the cluster)
        OUT: Download only (output data that flows from the cluster)
        BOTH: Bidirectional (can upload and download)
    """
    IN = "in"
    OUT = "out"
    BOTH = "both"
    
    @classmethod
    def _missing_(cls, value: str) -> "DirectoryType":
        """Support legacy/verbose names for backward compatibility."""
        aliases = {
            "input": cls.IN,
            "output": cls.OUT,
            "bidirectional": cls.BOTH,
        }
        if value.lower() in aliases:
            return aliases[value.lower()]
        return None


@dataclass
class DirectoryConfig:
    """Configuration for a directory mapping."""
    type: DirectoryType = DirectoryType.BOTH


@dataclass
class ContainerConfig:
    """Configuration for a container image."""
    file: str
    tag: str
    remote_path: Optional[str] = None
    stage: Optional[str] = None


@dataclass
class JobConfig:
    """Configuration for a job definition.
    
    Attributes:
        script: Path to the SLURM script file.
        env: Environment variables to inject (name -> value).
        time: Suggested time limit (informational, override via CLI).
        nodes: Suggested node count (informational, override via CLI).
        gpus_per_node: Suggested GPUs per node (informational).
        cpus_per_task: Suggested CPUs per task (informational).
    """
    script: str
    env: dict[str, str] = field(default_factory=dict)
    time: Optional[str] = None
    nodes: Optional[int] = None
    gpus_per_node: Optional[int] = None
    cpus_per_task: Optional[int] = None


@dataclass
class WorkdirConfig:
    """Configuration for workdir mapping."""
    remote: str
    local: str = "."


@dataclass
class FcwConfig:
    """Main configuration object."""
    project: str = "default"
    workdir: WorkdirConfig = field(default_factory=lambda: WorkdirConfig(remote="", local="."))
    directories: dict[str, DirectoryConfig] = field(default_factory=dict)
    containers: dict[str, ContainerConfig] = field(default_factory=dict)
    jobs: dict[str, JobConfig] = field(default_factory=dict)
    
    # Resolved at runtime
    _config_path: Optional[Path] = field(default=None, repr=False)

    def resolve_path(self, path: str, remote: bool = True) -> str:
        """Resolve a relative path against workdir."""
        if path.startswith("/"):
            return path
        base = self.workdir.remote if remote else self.workdir.local
        return os.path.join(base, path)
    
    def get_directory_type(self, path: str) -> DirectoryType:
        """Get the type of a directory, defaulting to bidirectional."""
        # Normalize path for lookup
        path = path.strip("/")
        
        # Check exact match first
        if path in self.directories:
            return self.directories[path].type
        
        # Check if path is under a configured directory
        for dir_path, dir_config in self.directories.items():
            if path.startswith(dir_path.rstrip("/") + "/"):
                return dir_config.type
        
        return DirectoryType.BOTH
    
    def can_upload(self, path: str) -> bool:
        """Check if uploading to path is allowed."""
        dir_type = self.get_directory_type(path)
        return dir_type in (DirectoryType.IN, DirectoryType.BOTH)
    
    def can_download(self, path: str) -> bool:
        """Check if downloading from path is allowed."""
        dir_type = self.get_directory_type(path)
        return dir_type in (DirectoryType.OUT, DirectoryType.BOTH)


def expand_env_vars(value: str) -> str:
    """Expand environment variables in a string.
    
    Supports ${VAR} and ${VAR:-default} syntax.
    """
    def replace(match: re.Match) -> str:
        var_expr = match.group(1)
        if ":-" in var_expr:
            var_name, default = var_expr.split(":-", 1)
            return os.environ.get(var_name, default)
        return os.environ.get(var_expr, match.group(0))
    
    return re.sub(r'\$\{([^}]+)\}', replace, value)


def expand_config_refs(value: str, config_data: dict[str, Any]) -> str:
    """Expand internal config references like ${workdir.remote}."""
    def replace(match: re.Match) -> str:
        ref_path = match.group(1)
        parts = ref_path.split(".")
        
        # Navigate the config structure
        current = config_data
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return match.group(0)  # Keep original if not found
        
        if isinstance(current, str):
            return current
        return match.group(0)
    
    return re.sub(r'\$\{([a-zA-Z_][a-zA-Z0-9_.]*)\}', replace, value)


def process_value(value: Any, config_data: dict[str, Any]) -> Any:
    """Process a config value, expanding variables and references."""
    if isinstance(value, str):
        # First expand env vars, then config refs
        value = expand_env_vars(value)
        value = expand_config_refs(value, config_data)
        return value
    elif isinstance(value, dict):
        return {k: process_value(v, config_data) for k, v in value.items()}
    elif isinstance(value, list):
        return [process_value(v, config_data) for v in value]
    return value


def load_config(config_path: Optional[str | Path] = None) -> FcwConfig:
    """Load configuration from a YAML file.
    
    Args:
        config_path: Path to config file. If None, searches for fcw.yaml
                    in current directory, then ~/.fcw.yaml.
    
    Returns:
        Loaded and validated configuration.
    
    Raises:
        FileNotFoundError: If no config file found.
        ValueError: If config is invalid.
    """
    # Find config file
    if config_path is None:
        search_paths = [
            Path.cwd() / "fcw.yaml",
            Path.cwd() / ".fcw.yaml",
            Path.home() / ".fcw.yaml",
            Path.home() / ".config" / "fcw" / "config.yaml",
        ]
        for path in search_paths:
            if path.exists():
                config_path = path
                break
    
    if config_path is None:
        # Return default config if no file found
        return FcwConfig()
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load raw YAML
    with open(config_path) as f:
        raw_data = yaml.safe_load(f) or {}
    
    # Process values (expand env vars and refs)
    data = process_value(raw_data, raw_data)
    
    # Build config object
    config = FcwConfig(
        project=data.get("project", "default"),
        _config_path=config_path,
    )
    
    # Parse workdir
    if "workdir" in data:
        wd = data["workdir"]
        config.workdir = WorkdirConfig(
            remote=wd.get("remote", ""),
            local=wd.get("local", "."),
        )
    
    # Parse directories
    if "directories" in data:
        for path, dir_data in data["directories"].items():
            if isinstance(dir_data, dict):
                dir_type = DirectoryType(dir_data.get("type", "both"))
            else:
                dir_type = DirectoryType.BOTH
            config.directories[path] = DirectoryConfig(type=dir_type)
    
    # Parse containers
    if "containers" in data:
        for name, cont_data in data["containers"].items():
            config.containers[name] = ContainerConfig(
                file=cont_data.get("file", ""),
                tag=cont_data.get("tag", ""),
                remote_path=cont_data.get("remote_path"),
                stage=cont_data.get("stage"),
            )
    
    # Parse jobs
    if "jobs" in data:
        for name, job_data in data["jobs"].items():
            config.jobs[name] = JobConfig(
                script=job_data.get("script", ""),
                env=job_data.get("env", {}),
                time=job_data.get("time"),
                nodes=job_data.get("nodes"),
                gpus_per_node=job_data.get("gpus_per_node"),
                cpus_per_task=job_data.get("cpus_per_task"),
            )
    
    return config


def generate_default_config() -> str:
    """Generate a default fcw.yaml template."""
    return '''\
# fcw configuration file
project: my-hpc-app

# Workdir mapping - all paths are relative to this
workdir:
  remote: /scratch/${USER}/my-project
  local: .

# Directory declarations with data flow type
# type: in (upload only), out (download only), both (bidirectional)
directories:
  data/raw:
    type: in
  data/processed:
    type: out
  outputs:
    type: out
  code:
    type: both
  images:
    type: in

# Container definitions
containers:
  app:
    file: ./env/Dockerfile
    tag: myapp:latest
    remote_path: images/myapp.sqsh

# Job definitions with environment
jobs:
  preprocess:
    script: slurm/preprocess.sh
    env:
      DATA_IN: data/raw
      DATA_OUT: data/processed

  train:
    script: slurm/train.sh
    time: "12:00:00"
    nodes: 1
    env:
      DATA_DIR: data/processed
      OUTPUT_DIR: outputs
      # CONFIG: provided via --set at submit time

  evaluate:
    script: slurm/evaluate.sh
    env:
      MODEL_DIR: outputs
'''
