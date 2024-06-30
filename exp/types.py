import typing


Conversation = typing.TypedDict("Conversation", {"from": typing.Literal["system", "human", "gpt"], "value": str})
Row = typing.TypedDict("Row", {"conversations": list[Conversation]})