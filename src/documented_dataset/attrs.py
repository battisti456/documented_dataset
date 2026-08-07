from typing import TypedDict, Unpack


class Attrs(TypedDict, total = False):
    units:str
    description:str
    long_name:str
    dtype:str

class attrs(dict):
    def __init__(self,**kwargs: Unpack[Attrs]):
        super().__init__(**kwargs)