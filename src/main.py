import os
import json
import sys

# Dynamically get the absolute path to the project root (2 levels up from src/)
WORKSPACE_DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if os.path.exists(WORKSPACE_DIRECTORY) and WORKSPACE_DIRECTORY not in sys.path:
    sys.path.append(WORKSPACE_DIRECTORY)

from initialize import console
from tools import get_tool_descriptions_string, ALL_TOOLS
from model import client, MODEL, TravelPlan, ItineraryAgent, ItineraryRevisionAgent
from utils.data import VACATION_INFO_DICT

# 1. INITIALIZE DATA

from initialize import vacation_info, weather_for_dates, activities_for_dates


# 2. PLAN ITINERARY

ITINERARY_AGENT_SYSTEM_PROMPT = f"""
You are an experience travel agent. You excel at putting together tailored travel schedule exceeding your customers'expectations.

## Task
Your task is to plan a detailed day-by-day itinerary for your customers.
1. Read carefully the travel requirements of your customers. Take good note of their interests, dates and budget constraints.
2. Follow these rules:
 2.1 There should be at least one activity per day. Choose the activities based on the weather, your customers' interests and constraints
 2.2 Outdoor-only activities should be avoided.
 2.3 For a given day, check the weather before planning outdoor activities.
 2.4 During rain, events should be chosen based on traveler interests.
 2.5 Your itinerary should not exceed the total budget.
3. Plan a day-by-day itinerary following these steps:
    3.1 The itinerary should start on the date of arrival and should end on the date of departure
    3.2 For each day, 
        3.2.1 check the weather forecast and the available activities. Keep activities matching weather conditions and travelers' interests.
        3.2.2 then check the activity schedules and plan activities accordingly.
        3.2.3 check the overall itinerary cost. Remove activities one-by-one to match budget as necessary so that to still maximize the program.
    3.3 In case of alternatives, select activities so that each traveler has at least one activity in the itinerary that matches their interests.

## Output Format

Respond using two sections (ANALYSIS AND FINAL OUTPUT) in the following format:

    ANALYSIS:
    Day-by-day travel summary schedule

    FINAL OUTPUT:

    ```json
    {TravelPlan.model_json_schema()}
    ```

## Context

Weather forecast:
{weather_for_dates}

Available activities:
{activities_for_dates}
"""

itinerary_agent = ItineraryAgent(client=client, model=MODEL, system_prompt=ITINERARY_AGENT_SYSTEM_PROMPT)

if os.path.exists("outputs/output.json"):
    with open("outputs/output.json", "r") as f:
        travel_plan_saved = json.loads(f.read())
        VACATION_INFO_saved = travel_plan_saved["vacation_info"]
        travel_plan_saved = TravelPlan.model_validate(travel_plan_saved["itinerary"])

    if VACATION_INFO_saved == VACATION_INFO_DICT:
        console.print("Travel Plan loaded from output.json", style="bold")
        console.print(travel_plan_saved.model_dump_json(indent=2), style="green")
        travel_plan_1 = travel_plan_saved
    else:
        console.print("Preparing Travel Plan...Please wait...", style="bold cyan")
        travel_plan_1 = itinerary_agent.get_itinerary(
            vacation_info=vacation_info,
            model=MODEL,  # Optionally, you can change the model here
        )

else:
    console.print("Preparing Travel Plan...Please wait...", style="bold cyan")
    travel_plan_1 = itinerary_agent.get_itinerary(
        vacation_info=vacation_info,
        model=MODEL,  # Optionally, you can change the model here
    )

# Save to file
with open("outputs/output.json", "w") as f:
    to_save = {
        "vacation_info": VACATION_INFO_DICT,
        "itinerary": travel_plan_1.model_dump(),  # or json.loads(...model_dump_json())
    }
    json.dump(to_save, f, indent=4, default=str)  # indent for readability
    console.print("\n")
    console.print("Travel Plan saved to output.json", style="bold")
    console.print("\n")

"""console.print("====================== RUNNING VALIDATIONS INITIAL ITINERARY BEFORE FEEDBACK ======================", style="bold yellow")
console.print("\n")

eval_results = get_eval_results(
    vacation_info=vacation_info,
    final_output=travel_plan_1,
    eval_functions=ALL_EVAL_FUNCTIONS_1,
    traveler_feedback=None,
)

eval_results.model_dump()"""

from utils.data import TRAVELER_FEEDBACK


console.print("\n")
console.print("Traveler Feedback: ", TRAVELER_FEEDBACK, style="bold cyan")
console.print("\n")
"""console.print("====================== RUNNING VALIDATIONS INITIAL ITINERARY AFTER FEEDBACK ======================", style="bold yellow")
console.print("\n")

eval_results = get_eval_results(
    vacation_info=vacation_info,
    final_output=travel_plan_1,
    eval_functions=ALL_EVAL_FUNCTIONS,
    traveler_feedback=TRAVELER_FEEDBACK,
)

eval_results.model_dump()"""



ITINERARY_REVISION_AGENT_SYSTEM_PROMPT = f"""
You are an experienced travel agent expert in improving an initial travel itinerary to incorporate traveler feedback in a multi-step process using tool calls and reasoning.

## Task
Your task is to improve the itinerary based on this traveler's feedback: "{TRAVELER_FEEDBACK}"

# PROCESS
1. Read carefully the traveler's feedback to identify the requested modifications.
2. Follow the process below to improve the itinerary: 
- You will use step-by-step reasoning by
    - THOUGHT the next steps to take to complete the task and what next tool call to take to get one step closer to the final answer
    - ACTION on the single next tool call to take
- You will always respond with a single THOUGHT/ACTION message of the following format:
    THOUGHT:
    First, you will reason about the problem and determine the next logical action to take.
    ACTION:
    Based on your thought process, you will call ONE of the available tools.
 IMPORTANT: Start your analysis by invoking the evaluation tool to check the proposed itinerary and identify potential issues to be corrected.
3. Submit Final Response
- As soon as you know the final answer, run the evaluation tool again to check that all criteria are met.
- If the evaluation is passed then call the `final_answer_tool` in an `ACTION` message to submit the final itinerary.

# IMPORTANT RULES
- There should be at least one activity per day. Choose the activities based on the weather, the traverlers' interests and constraints
- During rain, events should be chosen based on traveler interests.
- The itinerary should not exceed the total budget defined by the travelers.
- The itinerary should start on the date of arrival and should end on the date of departure
- For a given day, timing of activities should be compatible with one another
- Check the overall itinerary cost. Remove activities one by one to match budget so that it still maximizes the program.
- In case of alternatives, select activities so that each traveler has at least one activity in the itinerary that matches their interests.

## Available Tools

{get_tool_descriptions_string(ALL_TOOLS)}

You will not use any other tools.
As soon as you know the final answer, run the evaluation tool again, then call the `final_answer_tool`only if all criteria are met.


## Output Format

    THOUGHT:
    One-liner summary of your reasonning.

    ACTION:
    {{"tool_name": "[tool_name]", "arguments": {{"arg1": "value1", ...}}}}

## Context

Weather forecast:
{weather_for_dates}

VACATION_INFO:
{VACATION_INFO_DICT}

Expected output for the final Travel Plan:
{TravelPlan.model_json_schema()}
""" 

itinerary_revision_agent = ItineraryRevisionAgent(system_prompt=ITINERARY_REVISION_AGENT_SYSTEM_PROMPT)

travel_plan_2 = itinerary_revision_agent.run_react_cycle(
    original_travel_plan=travel_plan_1, 
    max_steps=15,
    model=MODEL,
    client=client,
)

console.print("\n")
console.print("====================== FINAL ITINERARY ======================", style="bold yellow")
console.print("\n")
console.print(travel_plan_2.model_dump_json(indent=2), style="green")
console.print("\n")

# Save to file
with open("outputs/revised_output.json", "w") as f:
    to_save = {
        "vacation_info": VACATION_INFO_DICT,
        "itinerary": travel_plan_2.model_dump(),  # or json.loads(...model_dump_json())
    }
    json.dump(to_save, f, indent=4, default=str)  # indent for readability
    console.print("\n")
    console.print("Travel Plan saved to revised_output.json", style="bold cyan")
    console.print("\n")