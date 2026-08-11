class _Dims:
    def _names(self):
        return [name for name in dir(self) if name[0] != '_']