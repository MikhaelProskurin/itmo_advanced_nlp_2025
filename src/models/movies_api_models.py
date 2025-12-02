from enum import Enum
from pydantic import BaseModel, Field, field_validator
from pydantic.types import PositiveFloat, PositiveInt

class KinoteatrRuEndpointTemplates(Enum):
    """Endpoint templates for Kinoteatr.ru web api"""

    movie_card_url: str = "https://kinoteatr.ru/film/{movie_name}/{city_name}"
    theater_schedule_url: str = "https://kinoteatr.ru/raspisanie-kinoteatrov/{city_name}/{theater_name}"
    theater_list_url: str = "https://kinoteatr.ru/raspisanie-kinoteatrov/{city_name}"

class Movie(BaseModel):
    """Abstract movie model for current data-source"""

    name: str = Field(description="Movie name")
    director: str = Field(description="Movie's director")
    description: str = Field(description="Movie short description")
    year: str = Field(description="Film's release year")
    country: str = Field(description="Film's release country")

    @field_validator("name", "director", "year", "country", mode="before")
    def validate_string_input(cls, value: str) -> str:
        return str(value).strip().lower()

class ScreeningSession(BaseModel):
    """The model of showing one particular movie"""

    price: PositiveFloat = Field(description="The price for the average ticket, rub")
    session_datetime: str = Field(description="Session start datetime")
    genre_tags: list[str] = Field(description="Cinema generes that according to the movie")
    runtime: PositiveInt = Field(description="The movie runtime lenght, m")
    movie: Movie = Field(description="Movie model with it's own metadata")

    @field_validator("price", mode="before")
    def process_price_value(cls, value: str) -> PositiveFloat:
        value = "".join(list(filter(str.isdigit, value)))
        return float(value)
    
    @field_validator("runtime", mode="before")
    def process_runtime_value(cls, value: str) -> PositiveInt:
        hours, minutes = [v for v in value.split() if v.isdigit()]
        return int(hours) * 60 + int(minutes)

class DailyBillboard(BaseModel):
    """
    The daily billboard for one particular cinema hall.
    Model contains several screning sessions
    """

    showing_date: str = Field(description="Date of the full daily showing")
    daily_showing: list[ScreeningSession] = Field(description="The list of different movies showing at the specified date")

class TheaterMetadata(BaseModel):
    """
    The model of cinema theater metadata.
    It contains additional information, such as nearest subway stations etc.
    """

    title: str = Field(description="Cinema theater title")
    title_russian: str = Field(description="Cinema theater russian title")
    address: str = Field(description="Theater address in current town")
    subway_station: list[str] = Field(description="Nearest subway stations")

    @field_validator("address", mode="before")
    def validate_string_input(cls, value: str) -> str:
        return str(value).strip().lower()
    
    @field_validator("subway_station", mode="before")
    def process_subway_input(cls, value: str) -> list[str]:
        if isinstance(value, list):
            return list(map(str.strip, value))
        
        return [v.strip() for v in value.split(";")]
