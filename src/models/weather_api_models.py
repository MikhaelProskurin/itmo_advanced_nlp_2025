from enum import Enum
from pydantic import BaseModel, Field, field_validator

class OpenWeatherApiEndpoints(Enum):
    """Endpoint templates for openweathermap forecasting api"""
    
    geocoding_endpoint: str = "http://api.openweathermap.org/geo/1.0/direct"
    forecast_endpoint: str = "http://api.openweathermap.org/data/2.5/forecast"

class OpenWeatherGeocodingResponse(BaseModel):
    """Model that describes geocoding-api response"""

    latitude: float = Field(
        description="The latitude of the found city", 
        examples=[59.93, 55.75]
    )
    longitude: float = Field(
        description="The longitude of the found city", 
        examples=[30.31, 37.61]
    )
    
class OpenWeatherForecastResponse(BaseModel):
    """
    Model that describes relevant weather forecast information.
    This model further used in multi-agent system.
    """

    temperature: float = Field(description="Temperature in Celsius")
    feels_like: float = Field(description="The human perception of weather temperature")
    pressure: int = Field(description="Atmospheric pressure on the sea level, hPa")
    humidity: int = Field(description="Humidity, %")

    weather_description: str = Field(description="Weather condition")
    cloud_persentage: int = Field(description="Cloudiness, %")
    wind_speed: float = Field(description="Wind speed, meter/sec")

    rain_volume: float = Field(description="Rain volume for last 3 hours, mm")
    
    timestamp: str = Field(
        description="Time of data forecasted",
        examples=["2022-09-04 12:00:00", "2025-11-10 00:00:00"]
    )

    @field_validator("rain_volume", mode="before")
    def process_rain_volume(cls, value) -> float:
        return value if value else 0.0
