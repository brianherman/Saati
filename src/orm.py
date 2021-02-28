from sqlalchemy import Column, Integer, String
from sqlalchemy.dialects.postgresql import ARRAY, UUID
from sqlalchemy.ext.declarative import declarative_base
from pydantic import BaseModel, constr

Base = declarative_base()


class EventOrm(Base):
    __tablename__ = 'EventLog'
    uuid = Column(UUID, primary_key=True, nullable=False, default=uuid.uuid4, unique=True)
    utterance_ts = 
    public_key = Column(String(20), index=True, nullable=False, unique=True)
    name = Column(String(63), unique=True)

class Event(BaseModel):
    uuid: str = uuid.uuid4()
    utterance_ts: datetime = datetime.now() 
    input: str
    output: List[str]
    sentiment: int
    sync_ratio: float
    interactions: int
