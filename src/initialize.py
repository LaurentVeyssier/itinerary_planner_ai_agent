from rich.console import Console
console = Console()
from model import VacationInfo
from utils.data import VACATION_INFO_DICT
from tools import call_weather_api_mocked, call_activities_api_mocked
import pandas as pd


# 1. GET DATA 

# Validate the VacationInfo data structure
vacation_info = VacationInfo.model_validate(VACATION_INFO_DICT)

# Get weather forecast for the vacation period
weather_for_dates = [
    call_weather_api_mocked(
        date=ts.strftime("%Y-%m-%d"), city=vacation_info.destination
    )
    for ts in pd.date_range(
        start=vacation_info.date_of_arrival,
        end=vacation_info.date_of_departure,
        freq="D",
    )
]

# Get activities for the vacation period
activities_for_dates = [
    activity
    for ts in pd.date_range(
        start=vacation_info.date_of_arrival,
        end=vacation_info.date_of_departure,
        freq="D",
    )
    for activity in call_activities_api_mocked(
        date=ts.strftime("%Y-%m-%d"), city=vacation_info.destination
    )
]
