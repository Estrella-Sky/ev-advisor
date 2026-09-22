"""Agent 可调用的工具集合。"""

from src.tools.calculator_tool import calc_landing_price
from src.tools.rag_tool import get_car_specs, search_car_knowledge
from src.tools.search_tool import web_search

ALL_TOOLS = [search_car_knowledge, get_car_specs, web_search, calc_landing_price]

__all__ = ["ALL_TOOLS", "calc_landing_price", "get_car_specs", "search_car_knowledge", "web_search"]

