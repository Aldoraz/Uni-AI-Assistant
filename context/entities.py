from typing import Literal
from dataclasses import dataclass

@dataclass
class Message:
    role: Literal["system", "user", "assistant"]
    content: str