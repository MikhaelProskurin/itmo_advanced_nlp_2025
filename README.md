# KinoteatrRu recomendations agent

Проект представляет из себя AI-агента, составляющего персонализированные рекомендации для клиенов хостинга кинозалов.
Агент выполняет роль планировщика и персонального ассистента, который помогает клиенту подобрать наиболее релевантный для себя прокат в каком-либо из кинотеатров хостинга.

## Project Description

*   **stack** -> langchain, langgraph, pydantic, aiohttp
*   **Цель** -> Повысить конверсию от посещений платформы **kinoteatr.ru**, через предоставление AI-рекомендаций и персональное планирование.
*   **Функционал** -> geocoding-api, weather-summary, cinema-suggestion, movie-suggestion.

## Usage

### Core requirements
*   Установленный [Python 3.10+](https://www.python.org/)
*   Перечень пакетов, описанных в requirements.txt проекта.

### Installation
1.  Клонируйте репозиторий:
    ```bash
    git clone https://github.com/MikhaelProskurin/itmo_advanced_nlp_2025.git
    cd <your_directory>
    ```

2.  Установите зависимости:
    ```bash
    pip install -r requirements.txt
    ```

3.  Настройте переменные окружения (при необходимости):
    *   Создайте файл ```.env```.
    *   Заполните необходимые ключи (LLM_API_KEY, BASE_URL, WEATHER_API_KEY) в созданном ```.env``` файле. 

    ```python
    LLM_API_KEY=... # (typical api-key for llm access)
    BASE_URL=... # (endpoint of llm-inference-server)
    WEATHER_API_KEY=... # (api.openweathermap.org)
    ```

### Примеры использования

Примеры использования агента находятся в ```app.py``` и ```example.ipynb```
