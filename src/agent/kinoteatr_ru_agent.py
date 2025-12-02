import os
import logging
from dotenv import find_dotenv, load_dotenv

from datetime import datetime

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RetryPolicy

from langchain_openai import ChatOpenAI

from langchain.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser

from agent.prompt_storage import NodesPromptStorage

from models.agent_input_output_models import (
    AgentWorkflowState,
    PreferencesNodeOutput,
    TheaterSuggestionNodeOutput,
    RouterNodeOutput
)
from models.movies_api_models import (
    TheaterMetadata,
    DailyBillboard
)
from tools.weather_api_tool import _provide_weather_forecast
from tools.web_parsing_tool import (
    _provide_theaters_information,
    _provide_showing_schedule
)


class KinoteatrRuAgent:
    """
    A specialized LLM-powered agent for personalized entertainment recommendations 
    integrating weather, theater, and movie data.

    The system implements a ReAct-based reasoning approach with conditional
    routing between tool execution and summarization pathways.
    """

    def __init__(self, model_name: str = "qwen3-32b", *args, **kwargs) -> None:
        
        load_dotenv(find_dotenv())
        self.logger = logging.getLogger()
        
        self.llm = ChatOpenAI(
            base_url=os.getenv("BASE_URL"),
            api_key=os.getenv("LLM_API_KEY"),
            model=model_name
        )

        toolkit = [
            _provide_weather_forecast,
            _provide_theaters_information,
            _provide_showing_schedule
        ]
        self.tools = {_tool.name: _tool for _tool in toolkit}

        self.prompts = NodesPromptStorage
        self.llm_with_tools = self.llm.bind_tools(toolkit)
        self.graph = StateGraph(AgentWorkflowState)
    
    @property
    def compiled_graph(self) -> CompiledStateGraph:
        """
        Compiles and returns execution graph for the recommendation system.
        This property builds a LangGraph state machine that orchestrates the flow of
        user requests through various processing nodes.
        """

        rp = RetryPolicy(initial_interval=0.5, max_attempts=2)

        self.graph.add_node(
            "preferences_extraction_node", 
            self.preferences_extraction_node, 
            retry_policy=rp
        )
        self.graph.add_node("router_node", self.router_node, retry_policy=rp)
        self.graph.add_node("executor_node", self.executor_node, retry_policy=rp)

        self.graph.add_node(
            "weather_summarization_node",
            self.weather_summarization_node,
            retry_policy=rp
        )
        self.graph.add_node(
            "theater_suggestion_node", 
            self.theater_suggestion_node, 
            retry_policy=rp
        )
        self.graph.add_node(
            "movies_suggestion_node",
            self.movies_suggestion_node,
            retry_policy=rp
        )
        self.graph.add_node(
            "summarization_node",
            self.summarization_node,
            retry_policy=rp
        )

        self.graph.add_edge(START, "preferences_extraction_node")
        self.graph.add_edge("preferences_extraction_node", "router_node")
        self.graph.add_conditional_edges(
            "router_node", 
            self.check_graph_routing, 
            {
                "summarization": "weather_summarization_node",
                "use_tools": "executor_node"
            }
        )
        self.graph.add_edge("executor_node", "router_node")

        self.graph.add_edge("weather_summarization_node", "theater_suggestion_node")
        self.graph.add_edge("theater_suggestion_node", "movies_suggestion_node")
        self.graph.add_edge("movies_suggestion_node", "summarization_node")
        self.graph.add_edge("summarization_node", END)

        return self.graph.compile()
    
    def check_graph_routing(self, state: AgentWorkflowState) -> AgentWorkflowState:
        """Re-Act cycle verificator for routing node."""
        return state["_routing"].routing_decision
        
    async def router_node(self, state: AgentWorkflowState) -> AgentWorkflowState:
        """
        Routes agent workflow based on user preferences and available data using ReAct reasoning.
        Processes the current state to determine optimal execution path through LLM analysis.

        Returns routing decision dictating whether to use tools or proceed with summarization.
        """
        
        parser = PydanticOutputParser(pydantic_object=RouterNodeOutput)

        fmt_kwargs = {
            "fmt": parser.get_format_instructions(),
            "theaters": state.get("_provide_theaters_information"),
            "weather": state.get("_provide_weather_forecast"),
            "preferences": state["_preferences"]
        }
        system_prompt = self.prompts.router_node.value.format(**fmt_kwargs)

        node_prompt = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=state["question"])
        ]

        model_response = await self.llm_with_tools.ainvoke(node_prompt)
        routing = parser.parse(model_response.content)

        return {"_routing": routing}

    async def preferences_extraction_node(self, state: AgentWorkflowState) -> AgentWorkflowState:
        """
        Extracts structured user preferences from natural language queries using LLM parsing.
        Analyzes the input question to identify location, date, and activity preferences.

        Returns parsed preferences to guide subsequent recommendation workflow steps.
        """

        parser = PydanticOutputParser(pydantic_object=PreferencesNodeOutput)

        system_prompt = (
            self.prompts
            .preferences_node.value
            .format(fmt=parser.get_format_instructions())
        )

        node_prompt = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=state["question"])
        ]

        model_response = await self.llm.ainvoke(node_prompt)
        preferences = parser.parse(model_response.content)

        return {"_preferences": preferences}
    
    async def executor_node(self, state: AgentWorkflowState) -> AgentWorkflowState:
        """
        Executes tool calls specified in routing decisions to fetch external data.
        Processes each requested tool asynchronously, collecting results for downstream use.
        """

        results = {}
        for call in state["_routing"].tool_calls:

            tool_result = await self.tools[call["name"]].ainvoke(call["args"])
            results[f"_{call['name']}"] = tool_result
        
        return results

    async def weather_summarization_node(self, state: AgentWorkflowState) -> AgentWorkflowState:
        """A node that summarizes weather forecast data"""

        system_prompt = (
            self.prompts
            .weather_summarization_node.value
            .format(weather=state["_provide_weather_forecast"])
        )
        node_prompt = [SystemMessage(content=system_prompt)]

        model_response = await self.llm.ainvoke(node_prompt)

        return {"weather_summary": StrOutputParser().parse(model_response.content)}
    
    async def theater_suggestion_node(self, state: AgentWorkflowState) -> AgentWorkflowState:
        """
        Generates concise weather summaries from raw forecast data using LLM processing.
        Transforms technical weather metrics into user-friendly recommendations and insights.

        Returns formatted summary suitable for integration with activity suggestions.
        """
        
        parser = PydanticOutputParser(pydantic_object=TheaterSuggestionNodeOutput)

        fmt_kwargs = {
            "theaters": state["_provide_theaters_information"],
            "preferences": state["_preferences"],
            "fmt": parser.get_format_instructions()
        }
        prompt_content = (
            self.prompts.theater_suggestion_node.value.format(**fmt_kwargs)
        )

        theater_suggestion_prompt = [SystemMessage(content=prompt_content)]

        model_response = await self.llm.ainvoke(theater_suggestion_prompt)

        return {"suggested_theater": parser.parse(model_response.content)}

    async def movies_suggestion_node(self, state: AgentWorkflowState) -> AgentWorkflowState:
        """
        Filters and recommends movie sessions based on user preferences and theater schedules.
        Retrieves daily showtimes, applies time-based filtering against user constraints.

        Returns personalized recommendations with highlighting optimal movie choices.
        """

        args = {
            "chosen_theater": state["suggested_theater"].theater.title,
            "city_name": state["_preferences"].city_name,
            "planned_date": state["_preferences"].planned_date
        }
        schedule: DailyBillboard = await self.tools["provide_showing_schedule"].ainvoke(args)

        filtered_sessions: list[TheaterMetadata] = []
        for session in schedule.daily_showing:

            _fmt_time = "%H:%M"

            user_preferences = datetime.strptime(state["_preferences"].user_time_preferences, _fmt_time)
            session_dttm = datetime.strptime(session.session_datetime, _fmt_time)

            if session_dttm >= user_preferences:
                filtered_sessions.append(session)

        fmt_kwargs = {
            "schedule": filtered_sessions,
            "preferences": state["_preferences"]
        }
        system_prompt = (
            self.prompts.movies_suggestion_node.value.format(**fmt_kwargs)
        )
        node_prompt = [SystemMessage(content=system_prompt)]

        model_response = await self.llm.ainvoke(node_prompt)

        return {"suggested_schedule": StrOutputParser().parse(model_response.content)}

    async def summarization_node(self, state: AgentWorkflowState) -> AgentWorkflowState:
        """
        Synthesizes final recommendations by summarizing all collected data into a cohesive response.
        Returns comprehensive answer ready for presentation to the user with optimized activity planning.
        """

        fmt_kwargs = {
            "suggested_theater": state["suggested_theater"].theater,
            "suggested_schedule": state["suggested_schedule"],
            "weather_forecast": state["weather_summary"],
        }
        prompt_content = (
            self.prompts.answer_summarization_node.value.format(**fmt_kwargs)
        )
        summarization_prompt = [SystemMessage(content=prompt_content)]

        model_response = await self.llm.ainvoke(summarization_prompt)
        
        return {"answer": StrOutputParser().parse(model_response.content)}
