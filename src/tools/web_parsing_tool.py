import re
import random
import aiohttp
from datetime import date

from typing import Sequence, Optional

from bs4 import BeautifulSoup

from langchain.tools import tool

from models.movies_api_models import (
    Movie,
    ScreeningSession,
    DailyBillboard,
    TheaterMetadata,
    KinoteatrRuEndpointTemplates
)

from models.agent_input_output_models import (
    ProvideTheatersToolSchema,
    ProvideScheduleToolSchema
)

from utils.utils import validate_city_name, DEFAULT_USER_AGENTS, SUPPORTABLE_CITIES_MAPPING

class AsyncKinoteatrRuScheduleParser:
    """Implementation of Kinoteatr.ru web-scrapper, that collects the data for LLM usage."""

    def __init__(self, city_name: str, user_agents: Sequence = None) -> None:
        
        self.city_name = city_name
        self._user_agents = user_agents or DEFAULT_USER_AGENTS
        self._api = KinoteatrRuEndpointTemplates
        
    @property
    def random_user_agent(self) -> str:
        """Returns random user agent from templated ones"""
        return random.choice(self._user_agents)
    
    async def parse_movie_card(self, session: aiohttp.ClientSession, formatted_url: str) -> Movie:
        """Scraps the data from movie card for screencast under specified city and movie code"""

        async with session.get(url=formatted_url, headers={"User-Agent": self.random_user_agent}) as response:
            
            if response.status == 200:

                content_tree = BeautifulSoup(await response.text(), "lxml")
                
                # main information container
                info = content_tree.find("div", class_="info")
        
                name, description, director = (
                    info.find("h1").contents[0],
                    info.find("p", {"itemprop": "description"}).contents[0],
                    info.find("span", {"itemprop": "director"})
                )

                additional_meta = info.find_all("p")[0].text

                # validating additional metadata
                try:
                    m = re.findall(r"\((.*)\)", additional_meta)[0]
                    year, country = m.split(",", maxsplit=1)

                except (Exception, ValueError) as ex:

                    print(additional_meta, ex)
                    year, country = None, None

                return Movie(
                    name=name, 
                    director=director.text if director else None,
                    description=description,
                    year=year, 
                    country=country
                )
        
    async def parse_daily_billboard(self, session: aiohttp.ClientSession, formatted_url: str, _date: str | date) -> DailyBillboard:
        """Scraps daily billboard of particular cinema hall in specified town"""

        if isinstance(_date, date): 
            _date = _date.strftime("%Y-%m-%d")

        async with session.get(url=formatted_url, headers={"User-Agent": self.random_user_agent}, params={"date": _date}) as response:
            
            if response.status == 200:

                content_tree = BeautifulSoup(await response.text(), "lxml")

                # schedule data container
                billboard_content = content_tree.find_all("div", class_="shedule_movie bordered gtm_movie")

                # to avoid api-requests overhead just collect movie models to map
                _movie_codes, daily_showing = {}, []

                for card in billboard_content:
                    
                    genres = card["data-gtm-ga4-list-item-genres"].split(", ")

                    runtime = card.find("div", class_="shedule_movie_description").find_all("span")[-1]
                    clear_runtime = str(runtime.contents[0]).strip()
                    
                    screening_sessions = card.find_all("div", class_="shedule_movie_sessions col col-md-8")

                    # collecting time, price and movies metadata
                    for screening in screening_sessions:

                        session_content = (
                            screening.find("span", class_="shedule_session_time"),
                            screening.find("span", class_="shedule_session_price"),
                            screening.find("a")["data-movie-code"]
                        )

                        # preprocess extracted values for escaping whitespaces
                        session_time, session_price, movie_code = map(
                            lambda tag: tag.strip() if isinstance(tag, str) else str(tag.contents[0]).strip(), 
                            session_content
                        )

                        # if we requests a move code before, there are no need to do it again
                        if movie_code not in _movie_codes:

                            movie = await self.parse_movie_card(
                                session, self._api.movie_card_url.value.format(movie_name=movie_code, city_name=self.city_name)
                            )
                            _movie_codes[movie_code] = movie

                        # building a model and collecting the results
                        session_model = ScreeningSession(
                            price=session_price,
                            session_datetime=session_time,
                            genre_tags=genres,
                            runtime=clear_runtime,
                            movie=_movie_codes.get(movie_code) or movie
                        )
                        daily_showing.append(session_model)

                return DailyBillboard(showing_date=_date, daily_showing=daily_showing)
    
    async def extract_theaters_meta(self, session: aiohttp.ClientSession, formatted_url: str) -> list[TheaterMetadata]:
        """Extracts data about cinema theaters in access for transmitted city"""

        async with session.get(url=formatted_url, headers={"User-Agent": self.random_user_agent}) as response:
            
            meta = []

            if response.status == 200:
                
                # parsing the content tree
                cinema_cards_list = (
                    BeautifulSoup(await response.text(), "lxml")
                    .find_all("div", class_="col-md-12")
                )
                
                # ectracting theater codes from tree
                for card in cinema_cards_list:
                    
                    link = card.find("a")["href"]

                    title, title_russian = (
                        link.split("/")[-2].strip(),
                        str(card.find("h3").contents[0]).strip()
                    )

                    additional_meta = card.find_all("span", class_="sub_title")

                    stripped_meta = [tag.text.strip() for tag in additional_meta]

                    # clearing the data from special chars
                    address, subway, *_ = [re.sub("\s{2,}", "; ", data) for data in stripped_meta]

                    # pydantic model creation
                    meta_model = TheaterMetadata(
                        title=title, 
                        title_russian=title_russian, 
                        address=address, 
                        subway_station=subway
                    )
                    meta.append(meta_model)

            return meta

@tool("provide_theaters_information", args_schema=ProvideTheatersToolSchema)
async def _provide_theaters_information(city_name: str) -> list[TheaterMetadata]:
    """
    A tool that allows you to get a theaters_information in the city.

    Args:
        city_name: The city from the user's preferences
    """
    
    valid_city_name = validate_city_name(
        value=city_name, 
        mapping=SUPPORTABLE_CITIES_MAPPING
    )

    kinoteatr_ru_parser = AsyncKinoteatrRuScheduleParser(city_name=valid_city_name)

    async with aiohttp.ClientSession() as session:
        
        _url = (
            kinoteatr_ru_parser._api
            .theater_list_url.value
            .format(city_name=valid_city_name)
        )

        theaters = await kinoteatr_ru_parser.extract_theaters_meta(
            session=session,
            formatted_url=_url
        )

        return theaters
    
@tool("provide_showing_schedule", args_schema=ProvideScheduleToolSchema)
async def _provide_showing_schedule(chosen_theater: str, city_name: str, planned_date: Optional[str] = None) -> DailyBillboard:
    """
    A tool that allows you to get a showing_schedule in specified theater.

    Args:
        chosen_theater: The most suitable cinema for the user
        city_name: The city_name from the user's preferences
        planned_date: The planned_date from the user's preferences
    """

    valid_city_name = validate_city_name(
        value=city_name, 
        mapping=SUPPORTABLE_CITIES_MAPPING
    )

    kinoteatr_ru_parser = AsyncKinoteatrRuScheduleParser(city_name=valid_city_name)

    async with aiohttp.ClientSession() as session:
        
        _url = (
            kinoteatr_ru_parser._api
            .theater_schedule_url.value
            .format(city_name=valid_city_name, theater_name=chosen_theater)
        )
        schedule = await kinoteatr_ru_parser.parse_daily_billboard(
            session=session, 
            formatted_url=_url, 
            _date=planned_date
        )
        return schedule
