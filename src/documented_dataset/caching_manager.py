from typing import Generic

from .types import DocumentedDatasetType


class Caching_Manager(Generic[DocumentedDatasetType]):  # noqa: UP046
    def __init__(self,cls:type[DocumentedDatasetType]):
        self.dds: type[DocumentedDatasetType] = cls
        self.flag = False
    def __enter__(self):
        self.flag = True
        return self
    def __exit__(self, *exc):
        self.flag = False
        return self
    def __bool__(self):
        match(self.dds._cache_policy):
            case "never":
                return True
            case "permitted":
                return self.flag
            case "always":
                return False