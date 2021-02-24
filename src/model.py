# generated by datamodel-codegen:
#   filename:  state.json
#   timestamp: 2021-02-24T17:29:01+00:00

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class ModelItem(BaseModel):
    responses: List[str]
    sentiment: float
    sync_ratio: float
    interactions: int
    instance_state: str = Field(..., alias='instance.state')
    request_time: Optional[str] = None


class Model(BaseModel):
    __root__: List[ModelItem]
