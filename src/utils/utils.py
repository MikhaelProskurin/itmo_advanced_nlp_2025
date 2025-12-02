SUPPORTABLE_CITIES_MAPPING = {
    "moscow": ("moscow", "москва"),
    "sankt-peterburg": ("saint-petersburg", "saint petersburg", "санкт-петербург", "санкт-петербург", "спб", "петербург")
}

SUPPORTABLE_WEATHER_FORECAST = {
    "Moscow": ("moscow", "москва"),
    "Saint-Petersburg": ("saint-petersburg", "saint petersburg", "санкт-петербург", "санкт-петербург", "спб", "петербург")
}

DEFAULT_USER_AGENTS = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36 OPR/123.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 8_8_2) AppleWebKit/601.38 (KHTML, like Gecko) Chrome/54.0.1713.307 Safari/601",
    "Mozilla/5.0 (compatible; MSIE 11.0; Windows NT 10.2; x64 Trident/7.0)"
)

def validate_city_name(value: str, mapping: dict) -> str:
    """Validates that given city_name is supportable by api"""

    clear_value = value.strip().lower()
    supported_key = [k for k, v in mapping.items() if clear_value in v][0]

    return supported_key
