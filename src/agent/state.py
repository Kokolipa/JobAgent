import uuid
from typing import Annotated, List

from langchain_core.messages import AnyMessage
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, HttpUrl
from typing_extensions import TypedDict


######################################
# <<<< Structured Outputs States >>>>
######################################
class UserInfo(BaseModel):
    "A model to collect information from a user"
    name: str = Field(description="The name of the user.")
    companies: List[str] = Field(description="A list containing the companies to conduct a research on")
    user_id: str = Field(
        description="Unique identifier of the user",
        default=f"USER-{uuid.uuid4()}"
    )


class SentimentReviews(BaseModel):
    "Sentiment and company overview summaries"
    companies: List[str] = Field(description="A list containing the companies names")
    positive_reviews: List[str] = Field(description="A list containing positive reviews about a companies")
    negative_reviews: List[str] = Field(description="A list containing negative reviews about a companies")


class CompaniesOverview(BaseModel):
    companies: List[str] = Field(description="A list containing the companies names")
    urls: List[HttpUrl] = Field(description="A list containing valid URL of company")
    companies_overview: List[str] = Field(description="A list containing a summary containing an overview for each of the companies")


#TODO: Add stuctured resume state

######################################
# <<<< State Definitions: Main >>>>
######################################
class AgentInputState(MessagesState):
    """InputState is only 'messages'."""


class ConversationState(TypedDict):
    """Represents the state of our conversation."""
    messages: Annotated[List[AnyMessage], add_messages] 
    name: str
    companies: List[str]
    sentiment_summaries: List[str]
    overview_summaries: List[str]
    final_report: str 
#NOTE: To formulate final report we need
    # (1.) Company overview
    # (2.) Sentiment Summary
    # (3.) Recent Role summary: Skills/ToolKit required for the role, role's essence, department info
    # (4.) Resume summarised into the following chunks: Intro-summary, Technical skills, Work Expeirience, Education & Certifications, Personal projects (if exist), Volunteering
    # (5.) Point 3 <> 4 will be compares such that given a overall summary for




#TODO: Add resume main states 



