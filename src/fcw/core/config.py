"""Configuration loading and validation."""

from __future__ import annotations

import os
import re
import warnings
from dataclasses import dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional

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
    stage: Optional[str] = None  # FIXME: What is the purpose of this if we have local_stages and remote_stage?
    toml: Optional[str] = None
    platform: Optional[str] = None
    build_args: Optional[Dict[str, str]] = None
    local_stages: Optional[List[str]] = None
    remote_stage: Optional[str] = None

    def get_local_stages(self) -> List[str]:
        """Return local stages, defaulting to ["download"]."""
        return self.local_stages or ["download"]

    def get_remote_stage(self) -> str:
        """Return remote stage, defaulting to "build-offline"."""
        return self.remote_stage or "build-offline"

    def stage_tag(self, stage: str) -> str:
        """Derive tag for a specific stage: <tag>-<stage>."""
        if ":" in self.tag:
            return f"{self.tag}-{stage}"
        return f"{self.tag}:{stage}"


@dataclass
class JobConfig:
    """Configuration for a job definition.

    Attributes:
        script: Path to the SLURM script file.
        env: Environment variables to inject (name -> value).
        time: SBATCH time limit (applied as override, CLI takes precedence).
        nodes: SBATCH node count (applied as override, CLI takes precedence).
        gpus_per_node: SBATCH GPUs per node (applied as override, CLI takes precedence).
        cpus_per_task: SBATCH CPUs per task (applied as override, CLI takes precedence).
    """
    script: str
    container: Optional[str] = None
    env: dict[str, str] = field(default_factory=dict)
    time: Optional[str] = None  # FIXME: all the 4 options below here are SBATCH options - why are they handled separately? Or asked differently, isn't there a generic recipe to capture SBATCH option overrides that carry over 1-to-1 without having to hardcode them in the config schema? Like what we do for gpus_per_node?
    nodes: Optional[int] = None
    gpus_per_node: Optional[int] = None
    cpus_per_task: Optional[int] = None

    # Fields that are not SBATCH options
    _NON_SBATCH_FIELDS: ClassVar[frozenset[str]] = frozenset({"script", "container", "env"})

    def sbatch_options(self) -> dict[str, str]:
        """Return non-None SBATCH fields as a dict suitable for overrides.

        Field names are converted from snake_case to kebab-case
        (e.g. ``gpus_per_node`` -> ``gpus-per-node``).
        """
        opts: dict[str, str] = {}
        for f in fields(self):
            if f.name in self._NON_SBATCH_FIELDS or f.name.startswith("_"):
                continue
            val = getattr(self, f.name)
            if val is not None:
                opts[f.name.replace("_", "-")] = str(val)
        return opts


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
    
    def resolve_container_image(self, cont: ContainerConfig) -> str:
        """Resolve full remote sqsh path: remote_path dir + tag-derived filename."""
        remote_dir = cont.remote_path or "ce-images/"
        sqsh_name = cont.tag.replace(":", "+").replace("/", "+") + ".sqsh"
        return os.path.normpath(self.resolve_path(os.path.join(remote_dir, sqsh_name), remote=True))

    def resolve_container_images_dir(self, cont: ContainerConfig) -> str:
        """Resolve the remote directory containing container images."""
        remote_dir = cont.remote_path or "ce-images/"
        return os.path.normpath(self.resolve_path(remote_dir, remote=True))

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
        # First expand env vars, then config refs # FIXME: is this the right order? Isn't the config ref syntax indistinguishable from env vars (and so, config refs would be interpreted as env vars, getting expanded mostly as empty strings or similar)?
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

    # Warn on unknown top-level keys
    known_keys = {"project", "workdir", "directories", "containers", "jobs"}
    unknown = set(data.keys()) - known_keys
    if unknown:
        warnings.warn(f"Unknown keys in {config_path}: {', '.join(sorted(unknown))}")

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
                stage=cont_data.get("stage"),  # FIXME: What is the purpose of this if we have local_stages and remote_stage?
                toml=cont_data.get("toml"),
                platform=cont_data.get("platform"),
                build_args=cont_data.get("build_args"),
                local_stages=cont_data.get("local_stages"),
                remote_stage=cont_data.get("remote_stage"),
            )
    
    # Parse jobs
    if "jobs" in data:
        for name, job_data in data["jobs"].items():
            config.jobs[name] = JobConfig(
                script=job_data.get("script", ""),
                container=job_data.get("container"),
                env=job_data.get("env", {}),
                time=job_data.get("time"),  # FIXME: all the 4 options below here are SBATCH options - why are they handled separately? Or asked differently, isn't there a generic recipe to capture SBATCH option overrides that carry over 1-to-1 without having to hardcode them in the config schema?
                nodes=job_data.get("nodes"),
                gpus_per_node=job_data.get("gpus_per_node"),
                cpus_per_task=job_data.get("cpus_per_task"),
            )
    
    return config


def generate_default_config() -> str:  # FIXME: is this a good default with all the concrete directories, containers and jobs? Users can already copy an example if needed, why would they need to generate it as a default? They can interactively add directories, containers and jobs. See also generate_interactive_config().
    """Generate a default fcw.yaml template."""
    return '''\
# fcw configuration file
project: my-fcw-app

# Workdir mapping - all paths are relative to this
workdir:
  remote: ${FIRECREST_SCRATCH}/my-fcw-app
  local: .

# Directory declarations with data flow type
# type: in (upload only), out (download only), both (bidirectional)
directories:
  data/raw:
    type: in
  data/processed:
    type: out
  data/outputs:
    type: out
  config:
    type: in

# Container definitions
containers:
  app:
    file: ./env/Dockerfile
    tag: my-fcw-app:latest
    remote_path: ./ce-images/
    toml: ./env/container.toml    # optional, user-editable enroot environment

# Job definitions with environment
# container: references a container name from the containers section;
# at submit time, fcw inlines the TOML into the script and resolves the image path
jobs:
  preprocess:
    script: slurm/preprocess.sh
    container: app
    env:
      DATA_IN: data/raw
      DATA_OUT: data/processed

  train:
    script: slurm/train.sh
    container: app
    time: "12:00:00"
    nodes: 1
    env:
      DATA_DIR: data/processed
      OUTPUT_DIR: data/outputs
      CONFIG_DIR: config

  evaluate:
    script: slurm/evaluate.sh
    container: app
    env:
      MODEL_DIR: data/outputs
'''


def add_container_to_config(
    config_path: Path,
    name: str,
    container: ContainerConfig,
) -> None:
    """Append a new container entry to fcw.yaml via text insertion.

    Finds the ``containers:`` block and inserts the new entry at the end of it,
    before the next top-level key.  This avoids PyYAML round-trip serialization
    and preserves existing comments and formatting.

    Raises:
        ValueError: If ``containers:`` block is not found or *name* already exists.
    """
    lines = config_path.read_text().splitlines(keepends=True)

    # Ensure trailing newline so the last line is always terminated
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"

    # Find the containers: line
    containers_idx: Optional[int] = None
    for i, line in enumerate(lines):
        if re.match(r'^containers:\s*(#.*)?$', line.rstrip()):
            containers_idx = i
            break
    if containers_idx is None:
        raise ValueError("'containers:' block not found in config")

    # Check for duplicate name
    name_pattern = re.compile(rf'^  {re.escape(name)}:\s*(#.*)?$')
    for line in lines[containers_idx + 1:]:
        if name_pattern.match(line.rstrip()):
            raise ValueError(f"Container '{name}' already exists in config")
        # Stop at next top-level key
        if line.strip() and not line[0].isspace() and not line.startswith('#'):
            break

    # Find insertion point: end of containers block (before next top-level key)
    insert_idx = len(lines)
    for i in range(containers_idx + 1, len(lines)):
        line = lines[i]
        if line.strip() and not line[0].isspace() and not line.startswith('#'):
            insert_idx = i
            break

    # Build the YAML snippet
    snippet = f"  {name}:\n"
    snippet += f"    file: {container.file}\n"
    snippet += f"    tag: {container.tag}\n"
    if container.remote_path is not None:
        snippet += f"    remote_path: {container.remote_path}\n"
    if container.stage is not None:
        snippet += f"    stage: {container.stage}\n"
    if container.toml is not None:
        snippet += f"    toml: {container.toml}\n"
    if container.platform is not None:
        snippet += f"    platform: {container.platform}\n"
    if container.build_args:
        snippet += "    build_args:\n"
        for k, v in container.build_args.items():
            snippet += f"      {k}: {v}\n"
    if container.local_stages is not None:
        snippet += f"    local_stages: [{', '.join(container.local_stages)}]\n"
    if container.remote_stage is not None:
        snippet += f"    remote_stage: {container.remote_stage}\n"

    lines.insert(insert_idx, snippet)
    config_path.write_text("".join(lines))


# ---------------------------------------------------------------------------
# Round-trip YAML editing via ruamel.yaml
# ---------------------------------------------------------------------------

def _load_yaml_roundtrip(config_path: Path) -> tuple:
    """Load YAML with ruamel for round-trip editing.

    Returns:
        Tuple of (yaml_instance, data) where data is a CommentedMap.
    """
    from ruamel.yaml import YAML

    ryaml = YAML()
    ryaml.preserve_quotes = True
    with open(config_path) as f:
        data = ryaml.load(f)
    if data is None:
        from ruamel.yaml.comments import CommentedMap

        data = CommentedMap()
    return ryaml, data


def _save_yaml_roundtrip(config_path: Path, ryaml: Any, data: Any) -> None:
    """Write YAML back preserving comments and formatting."""
    from io import StringIO

    stream = StringIO()
    ryaml.dump(data, stream)
    config_path.write_text(stream.getvalue())


def add_directory_to_config(config_path: Path, path: str, dir_type: DirectoryType) -> None:
    """Add a directory entry to fcw.yaml using round-trip YAML editing.

    Creates the ``directories:`` section if it does not exist.

    Raises:
        ValueError: If *path* already exists in the directories section.
    """
    from ruamel.yaml.comments import CommentedMap

    ryaml, data = _load_yaml_roundtrip(config_path)
    if "directories" not in data:
        data["directories"] = CommentedMap()
    dirs = data["directories"]
    if path in dirs:
        raise ValueError(f"Directory '{path}' already exists in config")
    entry = CommentedMap()
    entry["type"] = dir_type.value
    dirs[path] = entry
    _save_yaml_roundtrip(config_path, ryaml, data)


def remove_directory_from_config(config_path: Path, path: str) -> None:
    """Remove a directory entry from fcw.yaml.

    Raises:
        ValueError: If *path* is not found in the directories section.
    """
    ryaml, data = _load_yaml_roundtrip(config_path)
    dirs = data.get("directories")
    if dirs is None or path not in dirs:
        raise ValueError(f"Directory '{path}' not found in config")
    del dirs[path]
    _save_yaml_roundtrip(config_path, ryaml, data)


def add_container_to_config_roundtrip(
    config_path: Path,
    name: str,
    container: ContainerConfig,
) -> None:
    """Add a container entry to fcw.yaml using round-trip YAML editing.

    Creates the ``containers:`` section if it does not exist.

    Raises:
        ValueError: If *name* already exists in the containers section.
    """
    from ruamel.yaml.comments import CommentedMap

    ryaml, data = _load_yaml_roundtrip(config_path)
    if "containers" not in data:
        data["containers"] = CommentedMap()
    conts = data["containers"]
    if name in conts:
        raise ValueError(f"Container '{name}' already exists in config")
    entry = CommentedMap()
    entry["file"] = container.file
    entry["tag"] = container.tag
    if container.remote_path is not None:
        entry["remote_path"] = container.remote_path
    if container.stage is not None:
        entry["stage"] = container.stage
    if container.toml is not None:
        entry["toml"] = container.toml
    if container.platform is not None:
        entry["platform"] = container.platform
    if container.build_args:
        entry["build_args"] = dict(container.build_args)
    if container.local_stages is not None:
        entry["local_stages"] = list(container.local_stages)
    if container.remote_stage is not None:
        entry["remote_stage"] = container.remote_stage
    conts[name] = entry
    _save_yaml_roundtrip(config_path, ryaml, data)


def remove_container_from_config(config_path: Path, name: str) -> None:
    """Remove a container entry from fcw.yaml.

    Raises:
        ValueError: If *name* is not found in the containers section.
    """
    ryaml, data = _load_yaml_roundtrip(config_path)
    conts = data.get("containers")
    if conts is None or name not in conts:
        raise ValueError(f"Container '{name}' not found in config")
    del conts[name]
    _save_yaml_roundtrip(config_path, ryaml, data)


def add_job_to_config(config_path: Path, name: str, job: JobConfig) -> None:
    """Add a job entry to fcw.yaml using round-trip YAML editing.

    Creates the ``jobs:`` section if it does not exist.

    Raises:
        ValueError: If *name* already exists in the jobs section.
    """
    from ruamel.yaml.comments import CommentedMap

    ryaml, data = _load_yaml_roundtrip(config_path)
    if "jobs" not in data:
        data["jobs"] = CommentedMap()
    jobs = data["jobs"]
    if name in jobs:
        raise ValueError(f"Job '{name}' already exists in config")
    entry = CommentedMap()
    entry["script"] = job.script
    if job.container is not None:
        entry["container"] = job.container
    if job.time is not None:  # FIXME: all the 4 options below here are SBATCH options - why are they handled separately? Or asked differently, isn't there a generic recipe to capture SBATCH option overrides that carry over 1-to-1 without having to hardcode them in the config schema?
        from ruamel.yaml.scalarstring import DoubleQuotedScalarString

        entry["time"] = DoubleQuotedScalarString(job.time)
    if job.nodes is not None:
        entry["nodes"] = job.nodes
    if job.gpus_per_node is not None:
        entry["gpus_per_node"] = job.gpus_per_node
    if job.cpus_per_task is not None:
        entry["cpus_per_task"] = job.cpus_per_task
    if job.env:
        entry["env"] = dict(job.env)
    jobs[name] = entry
    _save_yaml_roundtrip(config_path, ryaml, data)


def remove_job_from_config(config_path: Path, name: str) -> None:
    """Remove a job entry from fcw.yaml.

    Raises:
        ValueError: If *name* is not found in the jobs section.
    """
    ryaml, data = _load_yaml_roundtrip(config_path)
    jobs = data.get("jobs")
    if jobs is None or name not in jobs:
        raise ValueError(f"Job '{name}' not found in config")
    del jobs[name]
    _save_yaml_roundtrip(config_path, ryaml, data)


def generate_interactive_config(project: str, remote_workdir: str, local_workdir: str) -> str:  # FIXME: is this a good default with all the concrete directories, containers and jobs? Users can already copy an example if needed, why would they need to generate it as a default? At most, I'd see commented out sections with ... placeholders where concrete items need to be filled in (but actually this would be more suitable for the non-interactive default config). See also generate_default_config().
    """Generate an fcw.yaml template with the given project settings.

    Follows the conventions from the basic example: ce-images/ for container
    images, data/ prefix for data directories, config/ as type in.
    """
    return f'''\
# fcw configuration file
project: {project}

# Workdir mapping - all paths are relative to this
workdir:
  remote: {remote_workdir}
  local: {local_workdir}

# Directory declarations with data flow type
# type: in (upload only), out (download only), both (bidirectional)
directories:
  data/raw:
    type: in
  data/processed:
    type: out
  data/outputs:
    type: out
  config:
    type: in

# Container definitions
containers:
  app:
    file: ./env/Dockerfile
    tag: {project}:latest
    remote_path: ./ce-images/
    toml: ./env/container.toml    # optional, user-editable enroot environment

# Job definitions with environment
# container: references a container name from the containers section;
# at submit time, fcw inlines the TOML into the script and resolves the image path
jobs:
  preprocess:
    script: slurm/preprocess.sh
    container: app
    env:
      DATA_IN: data/raw
      DATA_OUT: data/processed

  train:
    script: slurm/train.sh
    container: app
    time: "12:00:00"
    nodes: 1
    env:
      DATA_DIR: data/processed
      OUTPUT_DIR: data/outputs
      CONFIG_DIR: config

  evaluate:
    script: slurm/evaluate.sh
    container: app
    env:
      MODEL_DIR: data/outputs
'''
