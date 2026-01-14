# Copyright 2025 OPPO
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import base64
import dataclasses
import io
import re
import logging

from browsergym.core.action.functions import calculate, focus, press, take_note
from browsergym.experiments.loop import extract_action_content
import numpy as np
import openai
from PIL import Image

from browsergym.core.action.highlevel import HighLevelActionSet
from browsergym.experiments import AbstractAgentArgs, Agent
from browsergym.utils.obs import flatten_axtree_to_str, flatten_dom_to_str, prune_html
import os
import base64
import dataclasses
import io
import logging

import numpy as np
import openai
from PIL import Image

from browsergym.core.action.highlevel import HighLevelActionSet
from browsergym.experiments import AbstractAgentArgs, Agent
from browsergym.utils.obs import flatten_axtree_to_str, flatten_dom_to_str, prune_html
import os
logger = logging.getLogger(__name__)
api_key = os.getenv("OPENAI_API_KEY")


def image_to_jpg_base64_url(image: np.ndarray | Image.Image):
    """Convert a numpy array to a base64 encoded image url."""

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image.mode in ("RGBA", "LA"):
        image = image.convert("RGB")

    with io.BytesIO() as buffer:
        image.save(buffer, format="JPEG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode()

    return f"data:image/jpeg;base64,{image_base64}"



PROGRESS_SUMMARY_PROMPT = """
**Role:**
You are a **Progress Summarization Agent**. Your role is to analyze the current execution context and generate a concise, actionable summary that helps a browser automation agent decide its next step toward achieving the user’s goal.

**Inputs Provided:**
- `goal`: The ultimate task objective the user wants to accomplish.
- `axtree_txt`: The accessibility tree of the current webpage, describing its structure and interactive elements.
- `screenshot`: A visual rendering of the current webpage **with overlaid DOM element markers** -- such as bounding boxes labeled with element IDs, indices, or accessibility names—that spatially align with the nodes in `axtree_txt`. This allows visual grounding of UI components referenced in the accessibility tree.
- `action_history`: Sequence of actions already performed by the browser agent.
- `previous_summary`: The progress summary from the previous step.

**Output Requirements:**
Generate a structured, succinct progress summary with **exactly three sections**:
- **✅ Current Progress**
    - State clearly which sub-goals or key steps (relative to `goal`) have been completed.
    - If no relevant actions have been taken yet, state: “No progress made yet.”
- **🔍 Current State Analysis**
    - Assess whether the current page is on the critical path toward `goal`.
    - Identify **relevant interactive elements** from `axtree_txt` that are essential for progressing (e.g., input fields, buttons, links). Include their identifiers (e.g., `id`, `name`, or visible text) and current state (e.g., enabled/disabled, filled/empty, visible/invisible).
    - If the agent appears off-track (e.g., error page, login wall, irrelevant content), explicitly flag this and suggest recovery (e.g., “Page seems unrelated to goal—consider navigating back or restarting.”).
- **➡️ Next-step Guidance**(Conditional Section)
    - **Only include this section if the browser agent has clearly deviated from the task goal or taken a counterproductive action.**
    - Provide 1–2 concrete, executable recommendations for the browser agent’s immediate next action.
    - Actions must reference specific, existing elements from the `axtree_txt` (e.g., “Click button with id='submit-order'”, “Type 'wireless headphones' into input field labeled 'Search'”).
    - If the `goal` has been fully achieved, state clearly: “✅ Task completed.” (This may appear in **Current Progress** instead.)

**Formatting Rules:**
- Always include `✅ Current Progress` and `🔍 Current State Analysis`.
- Include **`➡️ Next-step Guidance` only when deviation or error is detected.**
- Keep language precise, objective, and free of speculation.
- Base all conclusions strictly on the provided inputs—do not assume or hallucinate elements, states, or outcomes.

**Example Output:**
✅ Current Progress: Completed account login and arrived at the checkout page.

🔍 Current State Analysis: The marked screenshot shows a shipping address form (input field labeled “address-input”, currently empty) and a “Place Order” button (marker ID: “submit-btn”, enabled). The axtree confirms these elements are focusable and editable.
"""



class BroswerAgent(Agent):
    """A basic agent using OpenAI API, to demonstrate BrowserGym's functionalities."""

    def obs_preprocessor(self, obs: dict) -> dict:
        logger.info(f"obs['calculate_result'] on obs_preprocessor: {obs.get('calculate_result', None)}")

        return {
            "chat_messages": obs["chat_messages"],
            "screenshot": obs["screenshot"],
            "screenshot_som": obs["screenshot_som"],
            "goal": obs["goal"],
            "goal_object": obs["goal_object"],
            "last_action": obs["last_action"],
            "last_action_error": obs["last_action_error"],
            "open_pages_urls": obs["open_pages_urls"],
            "open_pages_titles": obs["open_pages_titles"],
            "active_page_index": obs["active_page_index"],
            # "axtree_txt": flatten_axtree_to_str_new(obs["axtree_object"]),
            "axtree_txt": flatten_axtree_to_str(obs["axtree_object"]),
            # "axtree_txt": translate_node_to_str(prune_tree(obs["node_root"], mode="node"), mode="concise"),
            "pruned_html": prune_html(flatten_dom_to_str(obs["dom_object"])),
        }

    def __init__(
        self,
        model_name: str,
        chat_mode: bool,
        demo_mode: str,
        use_html: bool,
        use_axtree: bool,
        use_screenshot: bool,
        mode: str = "bid",
        use_som: bool = False,
        nav_multipages: bool = False,
        tips_path: str | None = None,
        task_type: str = None,
        use_full_action_history: bool = True,
        action_history_limit: int = 5,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.chat_mode = chat_mode
        self.use_html = use_html
        self.use_axtree = use_axtree
        self.use_screenshot = use_screenshot
        self.mode = mode
        self.use_som = use_som
        self.nav_multipages = nav_multipages
        self.tips_path = tips_path
        self.task_type = task_type
        self.note_contents = []
        self.use_full_action_history = use_full_action_history
        self.action_history_limit = action_history_limit
        # if not (use_html or use_axtree):
        #     raise ValueError(f"Either use_html or use_axtree must be set to True.")

        self.openai_client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )

        if self.mode == "bid":
            subsets = ["webarena", "custom"]
            # subsets = ["chat", "tab", "nav", "bid", "infeas"]
        elif self.mode == "demo":
            subsets = ["chat", "tab", "nav", "coord", "infeas", "custom"]
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Supported modes are 'bid' and 'demo'.")
        self.action_set = HighLevelActionSet(
            # subsets=["chat", "bid", "infeas"],  # define a subset of the action space
            subsets=subsets, # allow the agent to also use x,y coordinates
            custom_actions=[take_note, calculate, press, focus],
            strict=False,  # less strict on the parsing of the actions
            multiaction=False,  # does not enable the agent to take multiple actions at once
            demo_mode=demo_mode,  # add visual effects
        )
        # use this instead to allow the agent to directly use Python code
        # self.action_set = PythonActionSet())

        self.action_history = []
        self.progress_summary_content = "No progress made yet."

    def extract_search_gitlab_bid(self, full_prompt_txt: str) -> str:
        """
        Extract the bid corresponding to the Search GitLab node in the Page Accessibility tree from full_prompt_txt
        
        Args:
            full_prompt_txt: Complete prompt text, including Page Accessibility tree
            
        Returns:
            str: The bid corresponding to Search GitLab, returns None if not found
            
        Example:
            Input: "[146] textbox 'Search GitLab'"
            Output: "146"
        """

        # Match pattern: [number] textbox 'Search GitLab'
        # Also match possible variants, such as "Search GitLab" or Search GitLab (without quotes)
        patterns = [
            r'\[(\d+)\]\s+textbox\s+[\'"]Search GitLab[\'"]',  # [146] textbox 'Search GitLab'
            r'\[(\d+)\]\s+textbox\s+Search GitLab',            # [146] textbox Search GitLab
        ]
        
        for pattern in patterns:
            match = re.search(pattern, full_prompt_txt, re.IGNORECASE)
            if match:
                bid = match.group(1)
                logger.info("Found Search GitLab bid in full_prompt_txt: {}".format(match.group(0)))
                logger.info(f"Found Search GitLab bid: {bid}")
                return bid        
        logger.warning("Could not find Search GitLab bid in full_prompt_txt")
        return None

    def get_action(self, obs: dict) -> tuple[str, dict]:
        """
        Construct prompt and call OpenAI API to get the next action (non-chat mode)
        """
        if len(self.action_history) > 0:
            last_action = extract_action_content(self.action_history[-1])
            if last_action is None:
                logger.warning("Last action extract is None, skipping the action result history, will not calculate the result")
                logger.warning("Last action: {}".format(self.action_history[-1]))
            else:
                last_action = last_action.replace("```", "").strip()
                if 'calculate' in last_action:
                    def calculate(expression: str) -> float:
                        result = eval(expression)
                        return f"The result of the expression {last_action} is {result:.2f}"
                    action_result = eval(last_action)
                    self.action_history[-1] += f"\nAction result: {action_result}"
                elif 'take_note' in last_action:
                    note_content = last_action.replace("```", "").replace("take_note(", "").replace(")", "").strip()
                    self.note_contents.append(note_content)
                logger.info("Generating progress summary...")
                self.progress_summary(obs)
                logger.info("Progress summary generated:\n{}".format(self.progress_summary_content))

        # 1. System message (including action space and next step instruction)
        system_msgs = self.get_system_msgs(obs)

        # 2. User message (non-chat)
        user_msgs = self.get_user_msgs_nonchat(obs)

        # 3. Generate complete text log for debugging (optional)
        prompt_text_strings = []
        for message in system_msgs + user_msgs:
            if message["type"] == "text":
                prompt_text_strings.append(message["text"])
            elif message["type"] == "image_url":
                image_url = message["image_url"]
                if isinstance(image_url, dict):
                    image_url = image_url["url"]
                if image_url.startswith("data:image"):
                    prompt_text_strings.append(f"image_url: {image_url[:30]}... (truncated)")
                else:
                    prompt_text_strings.append(f"image_url: {image_url}")
            else:
                raise ValueError(f"Unknown message type {repr(message['type'])}")
        full_prompt_txt = "\n".join(prompt_text_strings)
        # logger.info(full_prompt_txt)

        # 4. Call OpenAI API
        response = self.openai_client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_msgs},
                {"role": "user", "content": user_msgs},
            ],
        )
        action = response.choices[0].message.content

        is_gitlab_page = self.task_type == "gitlab"
        if not is_gitlab_page:
            is_gitlab_page = "gitlab" in obs['open_pages_urls'][0].lower() or '8023' in obs['open_pages_urls'][0].lower()
        if not is_gitlab_page:
            is_gitlab_page = "gitlab" in obs['open_pages_titles'][0].lower()

        if is_gitlab_page and "fill(" in action:
            # Extract the bid corresponding to Search GitLab from full_prompt_txt
            search_gitlab_bid = self.extract_search_gitlab_bid(full_prompt_txt)
            logger.info("==============Search GitLab bid=========: {}".format(search_gitlab_bid))

            import re
            def repl(match):
                keyword = match.group(1).strip().strip("'\"")
                # Convert spaces in keywords to URL encoding format
                import urllib.parse
                encoded_keyword = urllib.parse.quote_plus(keyword)
                url = f'{os.environ["WA_GITLAB"].rstrip("/")}/search?search={encoded_keyword}&nav_source=navbar'
                return f"goto('{url}')"

            if search_gitlab_bid:
                logger.info("==============Raw action=========: {}".format(action))
                action = re.sub(rf"fill\(['\"]{search_gitlab_bid}['\"],\s*['\"]([^'\"]+)['\"]\)", repl, action)
                logger.info("gitlab search action replaced as goto action: {}".format(action))

        # 5. Record action history and return
        self.action_history.append(action)
        return action, {"full_prompt": full_prompt_txt}

    def get_system_msgs(self, obs: dict) -> list[dict]:
        """
        Construct system prompt messages (basic mode), including:
        - Basic instructions
        - Action space description and examples
        - Next step action instruction
        """
        if self.mode == "bid":
            action_instruct = "If you want to perform a click action, please use the `click` function; if you want to perform an input action, please use `fill` function instead of the `type` function."
            act = "click"
            act2 = "fill"
            example_action = """
# Example action
## Example 1
<think>\nTo find reviewers who mention underwater photos, I should first navigate to the reviews section of the product page. The reviews tab with 12 reviews seems like the logical next step.\n</think>\n<action>\n```click(\"1910\")```\n</action>
## Example 2
<think>\nThe search field is the main tool available on this page for finding specific products since the navigation menu does not show a category for electronics or computer components. The user needs to enter a specific search term for SSD hard drives with a minimum of 1TB capacity. Since we've clicked on the search field previously to set focus, the next step would be to enter the desired search term into the search input field.\n</think>\n<action>\n```fill('610', 'SSD hard drive 1TB')```\n</action>

            """
        elif self.mode == "demo":
            action_instruct = "If you want to perform a click action, please use the `mouse_click` function instead of the `click` function; If you want to perform an input action, please first use mouse_click to click on the coordinates of the position where you need to input, then use the `keyboard_type` function instead of the `type` function."
            act = "mouse_click"
            act2 = "keyboard_type"
            example_action = """
# Example action
## Example 1
<think>\nI have an invalid action type error because I used the incorrect format for the action command. The correct action for clicking should use the `mouse_click` command with coordinates. To proceed with adding the product with the lowest per unit price to the shopping cart, I must first visit all tabs, compare prices, and execute the correct mouse click action on the \"Add to Cart\" button on the selected product tab.\n</think>\n<action>\n```mouse_click(894, 701, 'left')```\n</action>
## Example 2
<think>\nThe error indicates an issue with an invalid action type. To address this, I need to ensure the proper action and focus on typing the product name accurately. The earlier step required the correct application of the input command for typing in a search bar. Thus, a possible approach is to use an appropriate function like `keyboard_type('Jiffy Corn Muffin Cornbread Mix')` in alignment with allowed actions.\n</think>\n<action>\n```keyboard_type('Jiffy Corn Muffin Cornbread Mix')```\n</action>
            """
        assert obs["goal_object"], "The goal is missing."

        base_instructions = f"""\
# Instructions

You are a UI Assistant, your goal is to help the user perform tasks using a web browser. You can communicate with the user via a chat, to which the user gives you instructions and to which you can send back messages. You have access to a web browser that both you and the user can see, and with which only you can interact via specific commands.

Review the instructions from the user, the current state of the page and all other information to find the best possible next action to accomplish your goal. Your answer will be interpreted and executed by a program, make sure to follow the formatting instructions.

Your response must strictly follow this output format:

<think>
Explain your reasoning step-by-step here. Describe how you analyze the current page state and your plan to accomplish the user's goal.
</think>
<action>
Only a **SINGLE** action is allowed in this tag, which will be executed by the program.
</action>

Here:
- The <think> tag should contain a clear and detailed explanation of your reasoning process, including how you interpret the task, analyze the page, and decide on your next step.
- Inside the <action> tag, you MUST provide exactly one valid action string formatted as ``` ```. This represents the single action to be executed. The action must be one of the valid actions from the action space described below.
- You may receive historical thoughts and executed actions as context; consider them carefully to avoid repeating ineffective steps.
- Always take into account the current task, the latest page screenshot, and the history of your prior actions to make an informed decision.
- Valid action names include but are not limited to: {act}, {act2}, submit. Please refer to the action space details provided below for the complete list and examples.
- If previous actions did not elicit a response from the web page, avoid repeating the same action to prevent redundant or futile operations.
- DO NOT visit an external search engine or website such as google to find the answer, you should use the information provided in the task and the current page.
- {action_instruct}
- Please don't scroll the page, as the environment does not support it.


GENERAL TIPS/Benchmark Quirks (VERY IMPORTANT):
    1. It's MUCH easier to find something on a website when there's a search bar instead of having to scroll through the whole page trying to locate it.
    2. When asked to do a search task or information retrieval, it can be very useful to use FILTERS to narrow down the search results.
    3. There are a lot of information so it will take a very long time if you just keep scrolling trying to find it. Best way is to search or filter it!
    4. Examine screenshots thoroughly, especially keep an eye out for signs like this "-". These could be negative signs which are important.
    5. **When URLs/text are TRUNCATED with copy buttons**: If you see text that's cut off in the UI (e.g., "https://example.com/very-long-url..." with a copy button), click the copy button then PASTE it somewhere to see the full content:
        - Click the URL/address bar and press Ctrl+V
        - Or paste into a text field
        - Take a screenshot to see the complete untruncated text


MESSAGE/ANSWER FORMAT (VERY IMPORTANT):
    When you send message to user, follow these principles:

    1. **Match the format exactly**: If examples show "557m", use that exact format
    2. **Provide complete answers**: Include sufficient context for the answer to stand alone
    3. **Add reasoning when appropriate**: For questions requiring judgment (yes/no, status checks,
        comparisons), include brief context or reasoning alongside your answer
    4. **Be precise with terminology**: Use exact wording from the source when copying text
    5. When asked to return answer in MM:COUNT format, return like this: "January: 1". It expect MM to be the explicit name of the month NOT a number.
    6. When asked how much is spent return just the decimal. So if item costs $7.50 return "7.50" or if it costs $0 return "0"
    7. When asked for configuration return as 2x2 instead of 2*2.
    8. If multiple matching entries exist for an amount-based question, itemize each amount in your reasoning and ensure the finalAnswer string contains the combined total (e.g., sum of all matching refunds) that satisfies the query.




"""
# - If you cannot find the element that needs to be clicked in the current screenshot, you can scroll a suitable distance for a better view.
# - When something does not work as expected (due to various reasons such as timeout), sometimes a simple retry can solve the problem. You can check the information in the history error to determine whether the retry has successfully displayed the correct interface. If it has returned to normal, you can ignore the error information.
        if self.tips_path is not None:
            if isinstance(self.tips_path, str):
                tips_paths = [self.tips_path]
            else:
                tips_paths = self.tips_path
            wa_envs = {key: value.rstrip("/") for key, value in os.environ.items() if key.startswith("WA_")}
            task_tips = ""
            for tips_path in tips_paths:
                if os.path.exists(tips_path):
                    tips = open(tips_path, "r", encoding="utf-8").read()
                    tips = tips.format_map(wa_envs)
                    if len(tips_paths) > 1:
                        tip_name = tips_path.split("/")[-1].split(".")[0]
                        task_tips += f"\n## {tip_name} Task Tips\n{tips}\n"
                    else:
                        task_tips += tips
            base_instructions += f"\n\n# Task Tips\n{task_tips}\n"
        if self.nav_multipages:
            base_instructions += f"""\

# Homepage:
You are executing a task that may require navigating through multiple pages. 
If you need to visit other websites, you can use the homepage at {os.environ["WA_HOMEPAGE"].rstrip("/")}. It has a list of websites you can visit.
If you're looking to search for uncertain information or something less specific, such as the nearest locations, buildings in a particular area, or other similar queries, you can visit the Wikipedia website on the homepage for more detailed information.
"""
        action_space_text = f"""\
# Action Space
{self.action_set.describe(with_long_description=True, with_examples=True)}

{example_action}

"""
        next_action_instruction = """\
# Next action
You will now think step by step and produce your next best action. Reflect on your past actions, any resulting error message, and the current state of the page before deciding on your next action.
"""

        return [
            {"type": "text", "text": base_instructions},
            {"type": "text", "text": action_space_text},
            {"type": "text", "text": next_action_instruction},
        ]

    def get_user_msgs_nonchat(self, obs: dict) -> list[dict]:
        """
        Construct user messages, including:
        - Goal information
        - Currently open tabs
        - Optional axtree/html/screenshot (based on configuration)
        - Action history and error information
        """
        user_msgs = []

        # Goal
        user_msgs.append({"type": "text", "text": "# Goal\n"})
        user_msgs.extend(obs["goal_object"])

        # Tab information
        user_msgs.append({"type": "text", "text": "# Currently open tabs\n"})
        for i, (url, title) in enumerate(zip(obs["open_pages_urls"], obs["open_pages_titles"])):
            active_marker = " (active tab)" if i == obs["active_page_index"] else ""
            user_msgs.append({
                "type": "text",
                "text": f"Tab {i}{active_marker}\n  Title: {title}\n  URL: {url}\n"
            })

        # Optional content
        if self.use_axtree:
            user_msgs.append({
                "type": "text",
                "text": f"# Current page Accessibility Tree\n\n{obs['axtree_txt']}\n"
            })
        if self.use_html:
            user_msgs.append({
                "type": "text",
                "text": f"# Current page DOM\n\n{obs['pruned_html']}\n"
            })
        if self.use_screenshot:
            if self.use_som:
                user_msgs.append({
                    "type": "text",
                    "text": "# Current page Screenshot with SOM\n"
                })
                user_msgs.append({
                    "type": "image_url",
                    "image_url": {
                        "url": image_to_jpg_base64_url(obs["screenshot_som"]),
                        "detail": "auto",
                    }
                })
            else:
                user_msgs.append({"type": "text", "text": "# Current page Screenshot\n"})
                user_msgs.append({
                    "type": "image_url",
                    "image_url": {
                        "url": image_to_jpg_base64_url(obs["screenshot"]),
                        "detail": "auto",
                    }
                })

        # Action history and error information
        if self.action_history and self.use_full_action_history:
            user_msgs.append({"type": "text", "text": "# History of past actions\n"})
            for action in self.action_history[-self.action_history_limit:]:
                user_msgs.append({"type": "text", "text": f"\n{action}\n"})
            if obs.get("last_action_error"):
                user_msgs.append({
                    "type": "text",
                    "text": f"# Error message from last action\n\n{obs['last_action_error']}\n"
                })

        if self.note_contents:
            user_msgs.append({"type": "text", "text": "# Notes\n"})
            user_msgs.append({"type": "text", "text": f"{'\n'.join(self.note_contents)}\n\n"})

        if self.progress_summary_content:
            user_msgs.append({"type": "text", "text": f"# Progress Summary\n{self.progress_summary_content}\n"})

        return user_msgs

    def progress_summary(self, obs: dict) -> str:
        """
        Generate a summary of the progress of the task.
        """
        messages = [
            {"role": "system", "content": PROGRESS_SUMMARY_PROMPT},
        ]

        axtree_txt = obs.get("axtree_txt", "")
        # Limit length to avoid irrelevant lengthy content taking up context
        if isinstance(axtree_txt, str) and len(axtree_txt) > 4000:
            axtree_txt = axtree_txt[:4000] + "\n...[truncated]..."

        user_contents = [
            {"type": "text", "text": f"# goal\n{obs['goal']}\n"},
            {"type": "text", "text": f"# axtree_txt\n{axtree_txt}\n"},
            {"type": "text", "text": f"# screenshot\n"},
            {"type": "image_url", "image_url": {
                "url": image_to_jpg_base64_url(obs["screenshot_som"]),
                "detail": "auto",
            }},
            {"type": "text", "text": f"# action_history\n{'\n'.join(self.action_history)}\n"},
            {"type": "text", "text": f"# previous_summary\n{self.progress_summary_content}\n"},
        ]

        messages.append({"role": "user", "content": user_contents})
        response = self.openai_client.chat.completions.create(
            model=self.model_name,
            messages=messages
        )
        self.progress_summary_content = response.choices[0].message.content.strip()
        return self.progress_summary_content


@dataclasses.dataclass
class BroswerAgentArgs(AbstractAgentArgs):
    """
    This class is meant to store the arguments that define the agent.

    By isolating them in a dataclass, this ensures serialization without storing
    internal states of the agent.
    """

    model_name: str = "gpt-5"
    chat_mode: bool = False
    demo_mode: str = "off"
    use_html: bool = False
    use_axtree: bool = True
    use_screenshot: bool = False
    mode: str = "bid"
    use_som: bool = False
    nav_multipages: bool = False
    tips_path: str | None = None
    task_type: str | None = None
    use_full_action_history: bool = True
    action_history_limit: int = 5
    def make_agent(self):
        return BroswerAgent(
            model_name=self.model_name,
            chat_mode=self.chat_mode,
            demo_mode=self.demo_mode,
            use_html=self.use_html,
            use_axtree=self.use_axtree,
            use_screenshot=self.use_screenshot,
            mode=self.mode,
            use_som=self.use_som,
            nav_multipages=self.nav_multipages,
            tips_path=self.tips_path,  # type: ignore
            task_type=self.task_type,
            use_full_action_history=self.use_full_action_history,
            action_history_limit=self.action_history_limit,
        )
