from pydantic import BaseModel


class ContentSignals(BaseModel):
    text: str
    author_id: str
    context: str = ""
