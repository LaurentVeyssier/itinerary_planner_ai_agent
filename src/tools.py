import datetime
from utils.data import ACTIVITY_CALENDAR, WEATHER_FORECAST
from model import TravelPlan,Activity
from typing import List


def call_activities_api_mocked(
    date: str | None = None, city: str | None = None, activity_ids: list[str] | None = None
) -> list[dict[str, str | int]]:
    """Calls the mocked activities API to get a list of activities for a given date and city.

    This function simulates an API call to retrieve activities based on the provided date and city.

    Args:
        date: The date to get activities for. Must be in the format YYYY-MM-DD.
        city: The city to get activities for. Only "AgentsVille" is supported.
        activity_ids: A list of activity IDs to filter the results. If None, all activities for the date and city will be returned.

    Returns:
        A list of activities for the given date and city. Currently only returns activities
        for AgentsVille between 2025-06-10 and 2025-06-15.
    """

    # If the city is not AgentsVille, return an empty list
    if city and city != "AgentsVille":
        return []

    # Verify the date format
    if date:
        try:
            datetime.datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            print(f"Invalid date format: {date}")
            return []

    # If the date is not between 2025-06-10 and 2025-06-15, return an empty list
    if date and (date < "2025-06-10" or date > "2025-06-15"):
        print(f"Date {date} is outside the valid range (2025-06-10 - 2025-06-15)")
        return []

    activities = ACTIVITY_CALENDAR

    if date:
        activities = [event for event in activities if event["start_time"].startswith(date)]

    if activity_ids:
        activities = [event for event in activities if event["activity_id"] in activity_ids]

    if not activities:
        print(f"No activities found for {date} in {city}.")
    return activities


def call_activity_by_id_api_mocked(activity_id: str):
    """Calls the mocked activity API to get an activity by its ID.

    Args:
        activity_id: The ID of the event to retrieve.

    Returns:
        A dictionary containing the event details, or an empty dictionary if not found.
    """
    for event in ACTIVITY_CALENDAR:
        if event["activity_id"] == activity_id:
            return event
    print(f"Event with ID {activity_id} not found.")
    return None


def call_weather_api_mocked(date: str, city: str) -> dict[str, str | int]:
    """
    Returns the weather forecast for a given date and city.

    Args:
        date: The date to get weather for. Must be in the format YYYY-MM-DD.
        city: The city to get weather for.

    Returns:
        A dictionary containing the weather forecast for the given date and city.
    """

    # If the city is not AgentsVille, return an empty dictionary
    if city != "AgentsVille":
        return {}

    # Verify the date format
    try:
        datetime.datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        print(f"Invalid date format: {date}")
        return {}

    # If the date is not between 2025-06-10 and 2025-06-15, return an empty dictionary
    if date < "2025-06-10" or date > "2025-06-15":
        print(f"Date {date} is outside the valid range (2025-06-10 - 2025-06-15)")
        return {}

    return next(
        (forecast for forecast in WEATHER_FORECAST if forecast["date"] == date), {}
    )


def do_chat_completion(messages: list[dict[str, str]], model=None, client=None, **kwargs):
    """A simple wrapper around OpenAI's chat completion API.

    Args:
        messages: A list of messages to send to the chat completion API.

    Returns:
        str: The response from the chat completion API.

    Raises:
        openai.OpenAIError: If the chat completion API returns an error.
    """
    if client is None:
        raise ValueError("A valid OpenAI client must be provided.")
    
    if model is None:
        raise ValueError("A valid model must be provided.")

    if "response_format" not in kwargs:
        response = client.chat.completions.create(  # type: ignore
            model=model,
            messages=messages,  # type: ignore
            **kwargs,  # type: ignore
        )
    else:
        response = client.beta.chat.completions.parse(  # type: ignore
            model=model,
            messages=messages,  # type: ignore
            **kwargs,  # type: ignore
        )

    if hasattr(response, "error"):
        raise RuntimeError(
            f"OpenAI API returned an error: {str(response.error)}"
        )

    return response.choices[0].message.content


def get_tool_descriptions_string(fns):
    """Generates a tool description from a function's docstring.
    Args:
        fns (list): List of functions to generate descriptions for.
    Returns:
        str: A formatted string containing the function names and their descriptions."""
    resp = ""
    for fn in fns:
        function_name = fn.__name__
        function_doc = fn.__doc__ or "No description provided."

        resp += f"* `{function_name}`: {function_doc}\n"

    return resp


def calculator_tool(input_expression) -> float:
    """Evaluates a mathematical expression and returns the result as a float.

    Args:
        input_expression (str): A string containing a valid mathematical expression to evaluate.

    Returns:
        float: The result of the evaluated expression.

    Example:
        >>> calculator_tool("1 + 1")
        2.0
    """
    import numexpr as ne
    return float(ne.evaluate(input_expression))


def get_activities_by_date_tool(date: str, city: str = 'AgentsVille') -> List[dict]:
    """
    Returns all activities available on a given date and a city
    inputs:
    - date: a date with string format YYYY-MM-DD
    - city: a city (default = 'AgentsVille')

    outputs:
    - list of activities available in the city for selected date.
    Each activity is a dictionary with key,value pairs based on Activity class pydantic model
    """
    resp = call_activities_api_mocked(date=date, city=city)

    return [Activity.model_validate(activity).model_dump() for activity in resp]


def final_answer_tool(final_output: TravelPlan) -> TravelPlan:
    """Returns the final travel plan

    Args:
        final_output (TravelPlan): The final travel plan to return.

    Returns:
        TravelPlan: The final travel plan.
    """
    return final_output


def run_evals_tool(travel_plan: TravelPlan) -> dict:
    """Runs all evaluation tools on the provided travel plan and vacation info.

    Args:
        travel_plan (TravelPlan): The travel plan to evaluate.

    Returns:
        EvaluationResults: The results of the evaluations.
    """
    from evals import get_eval_results,ALL_EVAL_FUNCTIONS
    from initialize import vacation_info
    from utils.data import TRAVELER_FEEDBACK

    if isinstance(travel_plan, dict):
        travel_plan = TravelPlan.model_validate(travel_plan)

    resp = get_eval_results(
        vacation_info=vacation_info,
        final_output=travel_plan,
        eval_functions=ALL_EVAL_FUNCTIONS,
        traveler_feedback=TRAVELER_FEEDBACK,
    )
    return {
        # Show the success status and any failures
        "success": resp.success,
        "failures": resp.failures,
    }



ALL_TOOLS = [
    calculator_tool,
    get_activities_by_date_tool,
    run_evals_tool,
    final_answer_tool,
]