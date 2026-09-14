from pydantic import BaseModel


# What the supervisor hands a subagent: a focused task plus the names of the tools
# the subagent is allowed to use.
class SubagentRequest(BaseModel):
    task: str
    tool_names: list[str]


# The short payload a subagent returns to the supervisor. The supervisor records
# `result` as a tool_result block and continues its own loop.
class SubagentResult(BaseModel):
    result: str
