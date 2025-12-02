import os
import aiohttp
from typing import Literal, Optional
from dotenv import find_dotenv, load_dotenv
from datetime import datetime, timedelta

from langchain.tools import tool

from models.weather_api_models import (
    OpenWeatherGeocodingResponse,
    OpenWeatherForecastResponse,
    OpenWeatherApiEndpoints
)

from models.agent_input_output_models import ProvideWeatherForecastToolSchema

from utils.utils import validate_city_name, SUPPORTABLE_WEATHER_FORECAST


class AsyncOpenWeatherApiClient:
    """This class interacts with OpenWeather-api's and could recieve a different weather forecast data."""

    def __init__(
            self, 
            weather_api_key: str = None, 
            city_name: str = "Saint-Petersburg", 
            country_code: str = "RU",
            units: Literal["metric", "imperial", "standard"] = "metric",
        ) -> None:
        
        if not weather_api_key:
            load_dotenv(find_dotenv())
            self.weather_api_key = os.getenv("WEATHER_API_KEY")
        else:
            self.weather_api_key = weather_api_key

        self.city_name = city_name
        self.country_code = country_code
        self.units = units

        self._api = OpenWeatherApiEndpoints
    
    @staticmethod
    def search_closest_forecast(
        target: str, forecast: list[OpenWeatherForecastResponse], n_nearest: int = 3
    ) -> list[OpenWeatherForecastResponse]:
        """Searches for the closest forecasts to the target timestamp"""

        _fmt, _tgt_fmt = "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"
        target_dttm = datetime.strptime(target, _tgt_fmt)

        sorted_forecast = sorted(forecast, key=lambda _m: datetime.strptime(_m.timestamp, _fmt))

        closest_forecast = filter(
            lambda _m: datetime.strptime(_m.timestamp, _fmt) > target_dttm - timedelta(hours=3), sorted_forecast
        )

        return list(closest_forecast)[:n_nearest]

    async def get_city_coordinates(self, session: aiohttp.ClientSession) -> OpenWeatherGeocodingResponse:
        """Requests for city latitude and longitude coordinates"""
        
        params = {
            "city_name": self.city_name,
            "country_code": self.country_code,
            "q": self.city_name + "," + self.country_code,
            "appid": self.weather_api_key
        }

        async with session.get(url=self._api.geocoding_endpoint.value, params=params) as response:
            
            if response.status == 200:

                response_content = await response.json()

                # creating the model from response
                return OpenWeatherGeocodingResponse(
                    latitude=response_content[0]["lat"], 
                    longitude=response_content[0]["lon"]
                )
    
    async def get_weather_forecast(
            self, 
            session: aiohttp.ClientSession, 
            latitude: float = None, 
            longitude: float = None,
            num_days: int = 3
        ) -> list[OpenWeatherForecastResponse]:
        """Requesting OpenWeatherAPI about weather forecast on available amount of days"""

        if not latitude and not longitude:

            geocoding = await self.get_city_coordinates(session)
            latitude, longitude = geocoding.latitude, geocoding.longitude

        params = {
            "lat": latitude,
            "lon": longitude,
            "units": self.units,
            "cnt": 8 * num_days,
            "appid": self.weather_api_key
        }

        result = []

        async with session.get(url=self._api.forecast_endpoint.value, params=params) as response:
            
            if response.status == 200:

                response_content = await response.json()

                # collecting the data for each timestamp in response
                for item in response_content["list"]:

                    main_data = item["main"]

                    temperature, feels_like, pressure, humidity = (
                        main_data["temp"],
                        main_data["feels_like"], 
                        main_data["pressure"], 
                        main_data["humidity"]
                    )

                    weather_description, cloud_persentage, wind_speed, rain_volume, timestamp = (
                        item["weather"][0]["description"],
                        item["clouds"]["all"],
                        item["wind"]["speed"],
                        item.get("rain", {}).get("3h"),
                        item["dt_txt"]
                    )

                    hourly_forecast = OpenWeatherForecastResponse(
                        temperature=temperature,
                        feels_like=feels_like,
                        pressure=pressure,
                        humidity=humidity,
                        weather_description=weather_description,
                        cloud_persentage=cloud_persentage,
                        wind_speed=wind_speed,
                        rain_volume=rain_volume,
                        timestamp=timestamp
                    )
                    result.append(hourly_forecast)

            return result
    
@tool("provide_weather_forecast", args_schema=ProvideWeatherForecastToolSchema)
async def _provide_weather_forecast(
    city_name: str, 
    planned_date: Optional[str], 
    approx_time: Optional[str], 
    num_days: Optional[int] = 1
) -> list[OpenWeatherForecastResponse]:
    """
    A Tool that allows you to get a weather forecast for the near future

    Args:
        planned_date: Date for weather forecast it requested
        approx_time: Approximate requested session time by user, it might be chosen session
        num_days: A number for which the forecast will be made
    """

    valid_city_name = validate_city_name(
        value=city_name, 
        mapping=SUPPORTABLE_WEATHER_FORECAST
    )

    api_client = AsyncOpenWeatherApiClient(city_name=valid_city_name)

    now = datetime.now()
    _fmt_time, _fmt_date = "%H:%M", "%Y-%m-%d"
    
    # handling date and time user preferences from llm response
    _time = datetime.strptime(approx_time, _fmt_time).strftime(_fmt_time) if approx_time else datetime.strftime(now, _fmt_time)
    _date = datetime.strptime(planned_date, _fmt_date).strftime(_fmt_date) if planned_date else datetime.strftime(now, _fmt_date)

    async with aiohttp.ClientSession() as session:

        forecast = await api_client.get_weather_forecast(session=session, num_days=num_days)

        nearest_forecast = api_client.search_closest_forecast(
            target=_date + " " + _time, forecast=forecast, n_nearest=3
        )
        return nearest_forecast
