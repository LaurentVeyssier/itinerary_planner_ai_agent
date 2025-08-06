import os
import json
import datetime
from enum import Enum
from typing import List, Literal, Optional
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from utils.data import Interest
from json_repair import repair_json
from initialize import console

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class OpenAIModel(str, Enum):
    GPT_41 = "gpt-4.1"  # Strong default choice for development tasks, particularly those requiring speed, responsiveness, and general-purpose reasoning. 
    GPT_41_MINI = "gpt-4.1-mini"  # Fast and affordable, good for brainstorming, drafting, and tasks that don't require the full power of GPT-4.1.
    GPT_41_NANO = "gpt-4.1-nano"  # The fastest and cheapest model, suitable for lightweight tasks, high-frequency usage, and edge computing.

MODEL = OpenAIModel.GPT_41_MINI  # Default model for this project


class Traveler(BaseModel):
    """A traveler with a name, age, and list of interests.
    
    Attributes:
        name (str): The name of the traveler.
        age (int): The age of the traveler.
        interests (List[Interest]): A list of interests of the traveler.
    """
    name: str
    age: int
    interests: List[Interest]


class VacationInfo(BaseModel):
    """Vacation information including travelers, destination, dates, and budget.
    Attributes:
        travelers (List[Traveler]): A list of travelers.
        destination (str): The vacation destination.
        date_of_arrival (datetime.date): The date of arrival.
        date_of_departure (datetime.date): The date of departure.
        budget (int): The budget for the vacation in fictional currency units.
    """
    travelers: List[Traveler] = Field(..., description="list of travelers in the group", min_items=1)
    destination: str = Field(..., value='AgentsVille', description="the vacation destination")
    date_of_arrival: datetime.date = Field(..., description="the date of arrival", gt=datetime.date(2025,6,9))
    date_of_departure: datetime.date = Field(..., description="the date of departure", lt=datetime.date(2025,6,16))
    budget: int = Field(..., description="The budget for the vacation", gt=0)


class Weather(BaseModel):
    temperature: float
    temperature_unit: Literal["Celsius", "Farenheit"] = Field(...,value='Celsius')
    condition: str


class Activity(BaseModel):
    activity_id: str
    name: str
    start_time: datetime.datetime = Field(..., description="start time of the activity", gt=datetime.date(2025,6,9))
    end_time: datetime.datetime  = Field(..., description="end time of the activity", lt=datetime.date(2025,6,16))
    location: str = Field(...,value='AgentsVille')
    description: str
    price: int = Field(...,description="price of the activity", gt=0)
    related_interests: List[Interest]


class ActivityRecommendation(BaseModel):
    activity: Activity
    reasons_for_recommendation: List[str]


class ItineraryDay(BaseModel):
    date: datetime.date
    weather: Weather
    activity_recommendations: List[ActivityRecommendation]


class TravelPlan(BaseModel):
    city: str
    start_date: datetime.date
    end_date: datetime.date
    total_cost: int
    itinerary_days: List[ItineraryDay]


class ChatAgent:
    """A chat agent that interacts with OpenAI's API to facilitate conversations.

    This class manages chat history, formats system prompts, and handles
    communication with OpenAI's chat completion API. It provides methods to
    add messages, get responses, and maintain conversation context.

    Attributes:
        system_prompt_template (str): Template for the system prompt using {variable_name} placeholders.
    """
    system_prompt = "You are a helpful assistant."
    messages = []

    def __init__(self, name=None, system_prompt=None, client=None, model=None):
        self.name = name or self.__class__.__name__
        if system_prompt:
            self.system_prompt = system_prompt
        self.client = client
        self.model = model
        self.reset()

    def add_message(self, role, content):
        """Add a message to the chat history.

        Args:
            role (str): The role of the message ("system", "user", or "assistant").
            content (str): The content of the message.

        Raises:
            ValueError: If the role is not one of "system", "user", or "assistant".
        """
        if role not in ["system", "user", "assistant"]:
            raise ValueError(f"Invalid role: {role}")
        self.messages.append({"role": role, "content": content})
        if role == "system":
            console.print(f"\n{self.name} - System Prompt", style="bold")
            #console.print(content, style="blue")
        elif role == "user":
            console.print(f"\n{self.name} - User Prompt", style="bold")
            #console.print(content)
        elif role == "assistant":
            console.print(f"\n{self.name} - Assistant Response", style="bold")
            console.print(content, style="yellow")

    def reset(self):
        """Reset the chat history and re-initialize with the system prompt.
        This method clears all existing messages and adds the system prompt
        formatted with the template_kwargs.
        """
        from textwrap import dedent

        system_prompt = dedent(self.system_prompt).strip()

        # Clear previous messages and add the system prompt
        self.messages = []
        self.add_message(
            "system",
            system_prompt,
        )

    def get_response(self, add_to_messages=True, model=None, client=None, **kwargs):
        """Get a response from the OpenAI API.

        Args:
            add_to_messages (bool, optional): Whether to add the response to the chat history
            using the add_message method and the assistant role. Defaults to True.

        Returns:
            str: The response from the OpenAI API.


        """
        from tools import do_chat_completion

        response = do_chat_completion(
            messages=self.messages,
            model=model or self.model,
            client=client or self.client,
            **kwargs
        )
        if add_to_messages:
            self.add_message("assistant", response)
        return response

    def chat(self, user_message, add_to_messages=True, model=None, **kwargs):
        """Send a message to the chat and get a response.

        Args:
            user_message (str): The message to send to the chat.

        Returns:
            str: The response from the OpenAI API.
        """
        self.add_message("user", user_message)
        return self.get_response(add_to_messages=add_to_messages, model=model, **kwargs)


class ItineraryAgent(ChatAgent):
    """An agent that plans itineraries based on vacation information, weather, and activities."""

    def get_itinerary(self, vacation_info: VacationInfo, model: Optional[OpenAIModel] = None) -> TravelPlan:
        """Generates a travel itinerary based on the provided vacation information."""

        response = (self.chat(
            user_message=vacation_info.model_dump_json(indent=2),
            add_to_messages=False,
            model=model or self.model,
        ) or "").strip()

        console.print(f"\n{self.name} - Assistant Response", style="bold")
        console.print(response, style="yellow")

        # Parse the response
        json_text = response.strip()

        if "```json" in json_text:
            json_text = json_text.split("```json")[1].split("```")[0].strip()

        try:
            travel_plan = TravelPlan.model_validate_json(json_text)
            return travel_plan
        except Exception as e:
            print("Error validating the following text as TravelPlan JSON:")
            print(json_text)
            raise


class ItineraryRevisionAgent(ChatAgent):

    def get_observation_string(self, tool_call_obj) -> str:
        """Extracts the observation from the thought-action response."""

        from tools import ALL_TOOLS
        self.tools = ALL_TOOLS

        if "tool_name" not in tool_call_obj:
            return "OBSERVATION: No tool name specified."

        if "arguments" not in tool_call_obj:
            return "OBSERVATION: No arguments specified."

        # If the arguments are not a dictionary, state the error
        if not isinstance(tool_call_obj["arguments"], dict):
            return f"OBSERVATION: Arguments should be a dictionary, got {type(tool_call_obj['arguments'])} instead."

        # If the tool name is not a string, state the error
        if not isinstance(tool_call_obj["tool_name"], str):
            return f"OBSERVATION: Tool name should be a string, got {type(tool_call_obj['tool_name'])} instead."

        tool_name = tool_call_obj["tool_name"]
        arguments = tool_call_obj["arguments"]

        tool_fn = None

        for tool in self.tools:
            if tool.__name__ == tool_name:
                tool_fn = tool
                break

        if tool_fn is None:
            return f"OBSERVATION: Unknown tool name '{tool_name}' in action string."

        try:
            tool_response = tool_fn(**arguments)
            return f"OBSERVATION: Tool {tool_name} called successfully with response: {tool_response}"
        except Exception as e:
            return f"OBSERVATION: Error occurred while calling tool {tool_name}: {e}"

    def run_react_cycle(
        self, original_travel_plan: TravelPlan, max_steps: int = 10, model: Optional[OpenAIModel] = None, client = None,
    ) -> TravelPlan:
        """Runs the ReAct cycle to revise the itinerary based on the evaluation results."""

        console.print(">>> ReACT cycle started", style="bold cyan")
        # Provide the original travel plan to revise
        self.add_message(
            role="user",
            content=f"Here is the itinerary for revision:\n{original_travel_plan.model_dump_json(indent=2)}",
        )
        resp = None

        # Run the ReAct cycle for a maximum number of steps
        for step in range(max_steps):
            console.print(f"\n======================== Step {step + 1} ========================", style="bold red")  #..........................
            # Get the thought-action response from the agent
            resp = self.get_response(model=model, client=client) or ""

            # If there is no action, report it to the LLM and continue
            if "ACTION:" not in resp:
                self.add_message(role="user", content="No action found in response.")
                continue

            action_string = resp.split("ACTION:")[1].strip()

            # Parse the tool call JSON from the action string
            try:
                # Fix any JSON formatting issues. e.g. missing closing braces, etc.
                action_string = repair_json(action_string)
                tool_call_obj = json.loads(action_string)
            except json.JSONDecodeError:
                print(f"Invalid JSON in action string: {action_string}")
                self.add_message(
                    role="user",
                    content=f"Invalid JSON in action string: {action_string}",
                )
                continue

            tool_name = tool_call_obj.get("tool_name", None)

            # If the final answer tool is called, validate and return the final travel plan
            if tool_name == "final_answer_tool":
                try:
                    new_travel_plan = TravelPlan.model_validate(
                        tool_call_obj["arguments"].get("final_output", tool_call_obj["arguments"])
                    )
                    console.print(">>> ReACT cycle completed. Returning final plan", style="bold cyan")
                    return new_travel_plan

                except Exception as e:
                    self.add_message(
                        role="user", content=f"Error validating final answer: {e}"
                    )
                    console.print(f"!!! final itinerary Validation failed: {e}", style="bold red")
                    continue

            # For all other tools, execute the tool call and add the observation
            else:
                observation_string = self.get_observation_string(
                    tool_call_obj=tool_call_obj
                )
                self.add_message(role="user", content=observation_string)

        raise RuntimeError(
            f"ReAct cycle did not complete within {max_steps} steps. Last response: {resp}"
        )