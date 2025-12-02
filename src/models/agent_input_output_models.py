from __future__ import annotations

import re
from datetime import datetime

from pydantic import BaseModel, Field, field_validator
from typing import Literal, Optional, Annotated, TypedDict

from langchain.messages import ToolCall

from .movies_api_models import TheaterMetadata
from .weather_api_models import OpenWeatherForecastResponse

class AgentWorkflowState(TypedDict):
    """State model for KinoteatrRuAgent"""

    question: str
    answer: Optional[str]
    suggested_theater: Optional[TheaterSuggestionNodeOutput]
    suggested_schedule: Optional[str]
    weather_summary: Optional[str]
    _routing: Optional[RouterNodeOutput]
    _preferences: Optional[PreferencesNodeOutput]
    _provide_weather_forecast: Optional[list[OpenWeatherForecastResponse]]
    _provide_theaters_information: Optional[list[TheaterMetadata]]

    # attribute for content reduction
    _reduction_state: Annotated[dict, lambda left, right: left | right]

class PreferencesNodeOutput(BaseModel):
    """Model that handles user preferences model extracted from LLM"""

    planned_date: Optional[str] = Field(
        description="The date when user planned to go to the cinema in '%Y-%m-%d' format",
        examples=["2025-01-01", "2025-12-12", None]
    )
    user_time_preferences: Optional[str] = Field(
        description="The time of day preferred by the user in '%H:%M' format", 
        examples=["15:30", "18:45"]
    )
    city_name: Optional[str]
    location: Optional[str]
    genres: Optional[list[str]]
    weather_forecast_required: Optional[bool]

    @field_validator("user_time_preferences", mode="before")
    def validate_time_preferences(cls, value: str) -> str:
        return re.search(r"\d{2}:\d{2}", value).group()
    
    @field_validator("planned_date", mode="before")
    def validate_planned_date(cls, value: str | None) -> str:
        
        default_value = datetime.now()
        _fmt = "%Y-%m-%d"

        if value:

            clear_value = re.search(r"\d{4}-\d{2}-\d{2}", value).group()

            planned_date = (
                default_value.strftime(_fmt)
                if datetime.strptime(clear_value, _fmt) < default_value
                else clear_value
            )
            return planned_date
        
        return default_value.strftime(_fmt)
        
        
class RouterNodeOutput(BaseModel):
    """Model that handles Re-Act cycle of multi-agent system"""

    routing_decision: Literal["use_tools", "summarization"] = Field(description="routing decision keyword")
    tool_calls: Optional[list[ToolCall]]

class TheaterSuggestionNodeOutput(BaseModel):
    """Model for structured output from movie-theater-expert node"""
    suggestion: Optional[str]
    theater: Optional[TheaterMetadata]

class ProvideTheatersToolSchema(BaseModel):
    city_name: Optional[str]

class ProvideScheduleToolSchema(BaseModel):
    city_name: Optional[str]
    planned_date: Optional[str]
    chosen_theater: Optional[str]

class ProvideWeatherForecastToolSchema(BaseModel):
    city_name: Optional[str]
    planned_date: Optional[str]
    approx_time: Optional[str]
    num_days: Optional[int]
