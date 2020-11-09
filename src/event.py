from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel
import uuid

class Event(BaseModel):
    uuid: str = uuid.uuid4()
    username: str
    utterance_ts: datetime
    sentiment: decimal
