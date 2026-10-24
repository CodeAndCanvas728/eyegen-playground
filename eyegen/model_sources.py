"""One file of a model: where it comes from, and what it is called here.

A source is one of two things. It is a file in a repository — ``repo`` and
``file`` — which is downloaded when the model is pulled. Or it is a file
that is already on this machine — ``path`` — which is used where it is and
never fetched. A model the user already has is described by the second
form, and the two travel the same road from there.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class ModelSource:
    """One file of a model: where it comes from, and what it is called here.

    ``name`` is the file name on this side of the download. It defaults to the
    name in the repository, which is what most models want, but a model whose
    files are named differently downstream can say so. It is a file name and
    nothing else: a name with a path in it would write outside the directory
    the model belongs to.
    """

    repo: str = ""
    file: str = ""
    name: str = ""
    path: str = ""

    def __post_init__(self) -> None:
        if self.path:
            if self.repo or self.file:
                raise ValueError(
                    "a model source is either a file in a repository or a file on this "
                    "machine, not both"
                )
            _check_local_path(self.path)
            if not self.name:
                object.__setattr__(self, "name", Path(self.path).name)
            elif self.name != Path(self.path).name:
                _check_file_name(self.name)
            return
        if not self.repo or not self.file:
            raise ValueError("a model source needs a 'repo' and a 'file', or a 'path'")
        if not self.name:
            object.__setattr__(self, "name", Path(self.file).name)
        _check_file_name(self.name)

    @property
    def is_local(self) -> bool:
        """Whether the file is already on this machine."""
        return bool(self.path)

    @classmethod
    def from_dict(cls, data: Mapping) -> "ModelSource":
        """Build a source from its JSON form."""
        if not isinstance(data, Mapping):
            raise ValueError(f"a model source must be an object, got {type(data).__name__}")
        path = data.get("path")
        if path is not None:
            if not isinstance(path, str) or not path:
                raise ValueError("a model source's 'path' must be a non-empty string")
            name = data.get("name") or ""
            if not isinstance(name, str):
                raise ValueError("the 'name' of a local model file must be a string")
            return cls(path=path, name=name)
        repo = data.get("repo")
        file = data.get("file")
        if not repo or not isinstance(repo, str):
            raise ValueError("a model source needs a 'repo' (owner/name)")
        if not file or not isinstance(file, str):
            raise ValueError(f"a model source in '{repo}' needs a 'file'")
        name = data.get("name") or ""
        if not isinstance(name, str):
            raise ValueError(f"the 'name' of {repo}/{file} must be a string")
        return cls(repo=repo, file=file, name=name)

    def to_dict(self) -> dict:
        """Return the JSON form of this source."""
        if self.path:
            data = {"path": self.path}
            if self.name != Path(self.path).name:
                data["name"] = self.name
            return data
        data = {"repo": self.repo, "file": self.file}
        if self.name != Path(self.file).name:
            data["name"] = self.name
        return data


def _check_file_name(name: str) -> None:
    """Refuse a name that is not a plain file name.

    The name becomes a path under the model's own directory. A name with a
    directory in it would put the file somewhere else, and a name that
    starts with a dash would be read as an option.
    """
    if not name:
        raise ValueError("a model file needs a name")
    if name in (".", ".."):
        raise ValueError(f"'{name}' is not a file name")
    if "/" in name or "\\" in name:
        raise ValueError(
            f"'{name}' must be a file name, not a path: the file is written into "
            "the model's own directory"
        )
    if name.startswith("-"):
        raise ValueError(f"'{name}' must not start with a dash")
    if "\0" in name:
        raise ValueError("a file name cannot hold a null character")


def _check_local_path(path: str) -> None:
    """Refuse a path that does not name a file.

    A local source is used where it is, so the path is the file itself: it
    has to name something, and it cannot hold a null character.
    """
    if not path:
        raise ValueError("a model file needs a path")
    if "\0" in path:
        raise ValueError("a path cannot hold a null character")
    if Path(path).name in ("", ".", ".."):
        raise ValueError(f"'{path}' does not name a file")
