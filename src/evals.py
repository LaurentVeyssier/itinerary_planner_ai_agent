from pydantic import BaseModel
from typing import List
from tools import call_activity_by_id_api_mocked, do_chat_completion
from model import OpenAIModel
from model import VacationInfo, TravelPlan, Activity
from model import client, ChatAgent
from initialize import weather_for_dates, activities_for_dates, console

class AgentError(Exception):
    pass

class EvaluationResults(BaseModel):
    success: bool
    failures: List[str]
    eval_functions: List[str]


def get_eval_results(vacation_info, final_output, eval_functions, traveler_feedback=None) -> EvaluationResults:
    """
    Evaluates the final output of the itinerary agent against a set of evaluation functions.
    Args:
        vacation_info (VacationInfo): The vacation information used to generate the itinerary.
        final_output (TravelPlan): The final output from the itinerary agent.
        eval_functions (List[callable]): A list of evaluation functions to apply.
    Returns:
        EvaluationResults: An object containing the success status, any failures, and the names of the evaluation functions used.
    """

    if not isinstance(vacation_info, VacationInfo):
        raise ValueError("vacation_info must be an instance of VacationInfo")
    if not isinstance(final_output, TravelPlan):
        raise ValueError("final_output must be an instance of TravelPlan")
    if not isinstance(eval_functions, list) or not all(
        callable(fn) for fn in eval_functions
    ):
        raise ValueError("eval_functions must be a list of callable functions")
    
    eval_results = []
    for eval_fn in eval_functions:
        try:
            eval_fn(vacation_info, final_output, traveler_feedback)
        except AgentError as e:
            error_msg = str(e)
            console.print("Evaluation Error", style="bold")
            console.print(error_msg, style="red")
            console.print("\n\n")
            eval_results.append(error_msg)
    
    return EvaluationResults(
        success=len(eval_results) == 0,
        failures=eval_results,
        eval_functions=[fn.__name__ for fn in eval_functions],
    )


def eval_start_end_dates_match(vacation_info: VacationInfo, final_output: TravelPlan, traveler_feedback=None):
    """Verifies that the arrival and departure dates in vacation_info match the start and end dates in final_output.

    Args:
        vacation_info (dict): Contains the vacation details including arrival and departure dates
        final_output (dict): Contains the itinerary details including start and end dates

    Raises:
        AgentError: If either the arrival date doesn't match the start date or the departure date doesn't match the end date
    """
    if (
        vacation_info.date_of_arrival != final_output.start_date
        or vacation_info.date_of_departure != final_output.end_date
    ):
        raise AgentError(
            f"Dates do not match: {vacation_info.date_of_arrival} != {final_output.start_date} or {vacation_info.date_of_departure} != {final_output.end_date}"
        )

    if final_output.start_date > final_output.end_date:
        raise AgentError(
            f"Start date is after end date: {final_output.start_date} > {final_output.end_date}"
        )

    print(f"✅ Start and end dates match: {final_output.start_date} to {final_output.end_date}")


def eval_total_cost_is_accurate(vacation_info: VacationInfo, final_output: TravelPlan, traveler_feedback=None):
    """Verifies that the total cost stated in final_output matches the sum of all activity prices.

    Args:
        vacation_info (dict): Contains the vacation details
        final_output (dict): Contains the itinerary details including activities with prices and total cost

    Raises:
        AgentError: If the calculated total cost doesn't match the stated total cost
    """
    actual_total_cost = 0

    for itinerary_day in final_output.itinerary_days:
        for activity_recommendation in itinerary_day.activity_recommendations:
            actual_total_cost += activity_recommendation.activity.price

    stated_total_cost = int(final_output.total_cost)

    if actual_total_cost != stated_total_cost:
        raise AgentError(
            f"Stated total cost does not match calculated total cost: {actual_total_cost} != {stated_total_cost}"
        )
    
    else:
        print(f"✅ Total cost is accurate: {actual_total_cost} == {stated_total_cost}")
    

def eval_total_cost_is_within_budget(vacation_info: VacationInfo, final_output: TravelPlan, traveler_feedback=None):
    """Verifies that the total cost stated in final_output is within the budget specified in vacation_info.

    Args:
        vacation_info (dict): Contains the vacation details including budget
        final_output (dict): Contains the itinerary details including total cost

    Raises:
        AgentError: If the total cost exceeds the budget
    """
    stated_total_cost = int(final_output.total_cost)
    if stated_total_cost > vacation_info.budget:
        raise AgentError(
            f"Total cost exceeds budget: {stated_total_cost} > {vacation_info.budget}"
        )

    else:
        print(f"✅ Total cost is within budget: {stated_total_cost} <= {vacation_info.budget}")


def eval_itinerary_events_match_actual_events(
    vacation_info: VacationInfo, final_output: TravelPlan, traveler_feedback=None
):
    """Verifies that the events listed in the itinerary match the actual events

    Args:
        vacation_info (dict): Contains the vacation details including traveler information and their interests
        final_output (dict): Contains the itinerary details including daily activities

    Raises:
        AgentError: If any traveler has no matching activities or if one traveler has more than twice
                   the number of matching activities compared to another traveler
    """

    event_ids_not_matching = []
    event_ids_missing = []

    for itinerary_day in final_output.itinerary_days:
        for activity_recommendation in itinerary_day.activity_recommendations:
            event_id = activity_recommendation.activity.activity_id
            # Assuming get_event_by_id is a function that retrieves the event by its ID

            reference_event = call_activity_by_id_api_mocked(event_id)

            if reference_event is None:
                event_ids_missing.append(event_id)

            elif Activity(**reference_event) != activity_recommendation.activity:
                print(
                    "---\n"
                    f"Event ID {event_id} does not match the reference event:\n"
                    f"Reference Event: {reference_event}\n"
                    f"Activity Event: {activity_recommendation.activity.model_dump()}"
                )
                event_ids_not_matching.append(event_id)
            else:
                # The event matches, so we can continue
                pass

    if event_ids_missing or event_ids_not_matching:
        raise AgentError(
            f"Event IDs missing: {event_ids_missing}\nEvent IDs not matching: {event_ids_not_matching}"
        )

    else:
        print(f"✅ Itinerary events match actual events")

def eval_itinerary_satisfies_interests(
    vacation_info: VacationInfo, final_output: TravelPlan, traveler_feedback=None
):
    """Verifies that the itinerary includes activities matching each traveler's interests.

    This function checks that each traveler has at least one activity in the itinerary that matches their interests.

        Args:
        vacation_info (dict): Contains the vacation details including traveler information and their interests
        final_output (dict): Contains the itinerary details including daily activities

    Raises:
        AgentError: If any traveler has no matching activities or if one traveler has more than twice
                   the number of matching activities compared to another traveler
    """
    traveler_to_interests = {}
    traveler_to_interest_hit_counts = {}

    for traveler in vacation_info.travelers:
        traveler_to_interests[traveler.name] = traveler.interests
        traveler_to_interest_hit_counts[traveler.name] = 0

    for traveler_name, interests in traveler_to_interests.items():
        for itinerary_day in final_output.itinerary_days:
            for activity_recommendation in itinerary_day.activity_recommendations:
                # Check if the activity matches any of the traveler's interests
                matching_interests = set(traveler_to_interests[traveler_name]) & set(
                    activity_recommendation.activity.related_interests
                )

                if matching_interests:
                    traveler_to_interest_hit_counts[traveler_name] += 1
                    print(
                        f"✅ Traveler {traveler_name} has a match with interest {matching_interests} at {activity_recommendation.activity.name}"
                    )

    travelers_with_no_interest_hits = [
        traveler
        for traveler, interest_hit_count in traveler_to_interest_hit_counts.items()
        if interest_hit_count == 0
    ]

    # If any of the travelers have 0 matches, raise an error
    if travelers_with_no_interest_hits:
        raise AgentError(
            f"Travelers {travelers_with_no_interest_hits} has no matches with the itinerary."
        )




ACTIVITY_AND_WEATHER_ARE_COMPATIBLE_SYSTEM_PROMPT = f"""
You are a skilled travel agent making sure that proposed activities are compatible with the weather.

## Task
You are given a proposed travel itinerary. Check that the weather is compatible with each recommended activity scheduled.
When there is not enough information, assume the activity IS_COMPATIBLE with the weather.
For outdoors activities during bad weather, answer IS_COMPATIBLE if an indoor backup option is mentionned in the detailed activity description (exemple: an outdoor activity description indicates an indoor backup in case of rain)

## Output format

    REASONING:
    one-liner summary explaining you reasoning.

    FINAL ANSWER:
    [IS_COMPATIBLE, IS_INCOMPATIBLE]

## Examples

Activity: "Trail Tales"
Activity type: Outdoor
Weather forecast: Rainy
REASONING:
The activity "Trail Tales" is outdoor while the weather is rainy. Therefore the activity is not compatible.
FINAL ANSWER:
IS_INCOMPATIBLE

---
Activity: "Soundtrack Picnic"
Activity type: Outdoor
Weather forecast: Rainy
REASONING:
The activity "Soundtrack Picnic" is outdoor while the weather is rainy. The activity description in the CONTEXT mentions an indoor backup. Therefore the activity is compatible.
FINAL ANSWER:
IS_COMPATIBLE

---
Activity: "Soundtrack Picnic"
Activity type: Outdoor
Weather forecast: Sunny
REASONING:
The activity "Soundtrack Picnic" is outdoor while the weather is sunny. Therefore the activity is compatible.
FINAL ANSWER:
IS_COMPATIBLE

---

## Context

Weather forecast:
{weather_for_dates}

Available activities details:
{activities_for_dates}

""".strip()


def eval_activities_and_weather_are_compatible(
    vacation_info: VacationInfo, final_output: TravelPlan, traveler_feedback=None
):
    """Verifies that no outdoor-only activities are scheduled during inclement weather conditions.

    Args:
        vacation_info (dict): Contains the vacation details
        final_output (dict): Contains the itinerary details including daily activities and weather conditions

    Raises:
        AgentError: If any outdoor activities are scheduled during weather conditions that could ruin them
    """ 

    activities_that_are_incompatible = []

    for itinerary_day in final_output.itinerary_days:
        weather_condition = itinerary_day.weather.condition

        for activity_recommendation in itinerary_day.activity_recommendations:
            resp = do_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": ACTIVITY_AND_WEATHER_ARE_COMPATIBLE_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": f"Activity: {activity_recommendation.activity.name}\nDescription: {activity_recommendation.activity.description}\nWeather Condition: {weather_condition}",
                    },
                ],
                client=client,
                # This is a high-frequency use case, so we use a fast and cheap model.
                model=OpenAIModel.GPT_41_NANO,
            )
    


            if ("IS_COMPATIBLE" in (resp or "")) or ("IS_COMPATIBILE" in (resp or "")):
                is_compatible = True
            elif ("IS_INCOMPATIBLE" in (resp or "")) or ("IS_INCOMPATIBILE" in (resp or "")):
                is_compatible = False
            else:
                raise RuntimeError(
                    f"Unexpected response from the model: {resp}. Expected 'IS_COMPATIBLE' or 'IS_INCOMPATIBLE'."
                )

            if is_compatible:
                print(
                    f"✅ Activity {activity_recommendation.activity.name} (on {itinerary_day.date}) and weather '{weather_condition}' are compatible."
                )

            else:
                activities_that_are_incompatible.append(
                    activity_recommendation.activity.name
                )
                print(
                    f"❌ Activity {activity_recommendation.activity.name} (on {itinerary_day.date}) and weather '{weather_condition}' are incompatible."
                )

    if activities_that_are_incompatible:
        raise AgentError(
            f"Activities that may be ruined by inclement weather: {activities_that_are_incompatible}"
        )


def eval_traveler_feedback_is_incorporated(
    vacation_info: VacationInfo, final_output: TravelPlan, traveler_feedback=None
):
    """Checks if the traveler's feedback was incorporated into the revised travel plan.

    Args:
        vacation_info (VacationInfo): The vacation information.
        final_output (TravelPlan): The revised travel plan.

    Raises:
        AgentError: If the traveler's feedback was not successfully incorporated.
    """

    agent = ChatAgent(
        system_prompt="""You are an expert in evaluating whether a travel plan incorporates traveler feedback.

    ## Output Format

    Respond using two sections (ANALYSIS AND FINAL OUTPUT) in the following format:

        ANALYSIS:
        * [step-by-step analysis]


        FINAL OUTPUT:
        [FULLY_INCORPORATED, PARTIALLY_INCORPORATED, NOT_INCORPORATED, or UNKNOWN]
        REASON: [reasoning for the final output]

    """,
        client=client,
        model=OpenAIModel.GPT_41,  # Use a powerful model for checking traveler feedback
    )

    resp = agent.chat(
        f"""Traveler Feedback: {traveler_feedback}
    Revised Travel Plan: {final_output.model_dump_json(indent=2)}
    """,
    )
    if "FINAL OUTPUT:" not in resp:
        raise RuntimeError(
            f"Unexpected response from the model: {resp}. Expected 'FINAL OUTPUT:'."
        )
    if "FULLY_INCORPORATED" not in resp:
        final_output = resp.split("FINAL OUTPUT:")[-1].strip()
        raise AgentError(
            f"Traveler feedback was not successfully incorporated into the revised travel plan. Response: {final_output}"
        )




ALL_EVAL_FUNCTIONS_1 = [
    eval_start_end_dates_match,
    eval_total_cost_is_accurate,
    eval_itinerary_events_match_actual_events,
    eval_itinerary_satisfies_interests,
    eval_total_cost_is_within_budget,
    eval_activities_and_weather_are_compatible,
]

ALL_EVAL_FUNCTIONS = ALL_EVAL_FUNCTIONS_1 + [eval_traveler_feedback_is_incorporated]
