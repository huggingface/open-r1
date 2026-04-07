from function_parser import FunctionCall
from copy import deepcopy

# from aguvis aguvis action space to custom action space:

# mobile.home() -> navigate_home()
# mobile.open_app(app_name='drupe') -> open_app(app_name: str) -> str:
# mobile.swipe(from_coord=[0.581, 0.898], to_coord=[0.601, 0.518]) -> swipe(from_coord=[0.581, 0.898], to_coord=[0.601, 0.518])
# mobile.back() -> navigate_back()
# mobile.long_press(x=0.799, y=0.911) -> long_press(x, y)
# mobile.terminate(status='success') -> final_answer(answer: str)

# answer('text') -> final_answer('text') OK
# mobile.wait(seconds=3) -> wait(seconds=3) OK
# pyautogui.hscroll(page=-0.1)
# ?
# pyautogui.scroll(page=-0.1) or pyautogui.scroll(0.13) OK
# -> negative: scroll(direction: Literal["up", "down"] = "up", amount: int = abs(page * 10))
# -> positive: scroll(direction: Literal["up", "down"] = "down", amount: int = abs(page * 10))
# pyautogui.click(x=0.8102, y=0.9463) -> click(x: int, y: int) OK
# pyautogui.doubleClick() -> double_click() OK
# pyautogui.hotkey(keys=['ctrl', 'c']) -> press(keys: str | list) OK
# pyautogui.press(keys='enter') or pyautogui.press(keys=['enter']) -> press(keys: str | list) OK
# pyautogui.moveTo(x=0.04, y=0.405) -> move_mouse(x: int, y: int) OK
# pyautogui.write(message='bread buns') -> type(text: str) OK
# pyautogui.dragTo(x=0.8102, y=0.9463) -> drag(x1, y1, x2, y2) OK but to recheck formatage in official dataset


def convert_to_pixel_coordinates(action: FunctionCall, resolution: tuple[int, int]) -> None:
    if "arg_0" in action.parameters:
        if isinstance(action.parameters["arg_0"], (list, tuple)):
            action.parameters["from_coord"] = (int(action.parameters["arg_0"][0] * resolution[0]), int(action.parameters["arg_0"][1] * resolution[1]))
        else:
            action.parameters["x"] = int(action.parameters["arg_0"] * resolution[0])
        del action.parameters["arg_0"]
    if "arg_1" in action.parameters:
        if isinstance(action.parameters["arg_1"], (list, tuple)):
            action.parameters["to_coord"] = (int(action.parameters["arg_1"][0] * resolution[0]), int(action.parameters["arg_1"][1] * resolution[1]))
        else:
            action.parameters["y"] = int(action.parameters["arg_1"] * resolution[1])
        del action.parameters["arg_1"]

def change_argument_name(action: FunctionCall) -> None:
    if "arg_0" in action.parameters:
        if isinstance(action.parameters["arg_0"], (list, tuple)):
            action.parameters["from_coord"] = (float(action.parameters["arg_0"][0]), float(action.parameters["arg_0"][1]))
        else:
            action.parameters["x"] = float(action.parameters["arg_0"])
        del action.parameters["arg_0"]
    if "arg_1" in action.parameters:
        if isinstance(action.parameters["arg_1"], (list, tuple)):
            action.parameters["to_coord"] = (float(action.parameters["arg_1"][0]), float(action.parameters["arg_1"][1]))
        else:
            action.parameters["y"] = float(action.parameters["arg_1"])
        del action.parameters["arg_1"]


def rename_parameters(action: FunctionCall) -> None:
    """
    Reorder FunctionCall parameters to use arg_0, arg_1, arg_2, etc. as keys.
    Preserves the order of the original parameters.
    
    Args:
        action: FunctionCall object to reorder parameters for
        
    """
    if not action.parameters:
        return
    
    for i, (key, value) in enumerate(deepcopy(action.parameters).items()):
        tmp = value
        del action.parameters[key]
        action.parameters[f"arg_{i}"] = tmp



def action_conversion(
    actions: list[FunctionCall], resolution: tuple[int, int]
) -> list[FunctionCall]:
    for i, action in enumerate(actions):
        rename_parameters(action)
        # MOBILE ACTIONS
        if action.function_name == "mobile.home":
            actions[i].function_name = "navigate_home"

        elif action.function_name == "mobile.open_app":
            actions[i].function_name = "open_app"

        elif action.function_name == "mobile.swipe":
            actions[i].function_name = "swipe"
            change_argument_name(actions[i])

        elif action.function_name == "mobile.back":
            actions[i].function_name = "navigate_back"

        elif action.function_name == "mobile.long_press":
            actions[i].function_name = "long_press"
            change_argument_name(actions[i])

        elif action.function_name in ["mobile.terminate", "answer"]:
            actions[i].function_name = "final_answer"

        elif action.function_name == "mobile.wait":
            actions[i].function_name = "wait"
            if "arg_0" in actions[i].parameters:
                actions[i].parameters["seconds"] = int(actions[i].parameters["arg_0"])
                del actions[i].parameters["arg_0"]

        # OS ACTION
        elif action.function_name == "pyautogui.click":
            actions[i].function_name = "click"
            change_argument_name(actions[i])

        elif action.function_name == "pyautogui.doubleClick":
            actions[i].function_name = "double_click"
            change_argument_name(actions[i])

        elif action.function_name == "pyautogui.rightClick":
            actions[i].function_name = "right_click"
            change_argument_name(actions[i])

        elif action.function_name in ["pyautogui.hotkey", "pyautogui.press"]:
            actions[i].function_name = "press"
            if "arg_0" in actions[i].parameters:
                actions[i].parameters["keys"] = actions[i].parameters["arg_0"]
                del actions[i].parameters["arg_0"]

        elif action.function_name == "pyautogui.moveTo":
            actions[i].function_name = "move_mouse"
            change_argument_name(actions[i])

        elif action.function_name == "pyautogui.write":
            actions[i].function_name = "type"

        elif action.function_name in ["pyautogui.scroll", "pyautogui.hscroll"]:
            arg_value = actions[i].parameters["arg_0"]
            if arg_value < 0:
                if action.function_name == "pyautogui.hscroll":
                    actions[i].parameters["direction"] = "left"
                else:
                    actions[i].parameters["direction"] = "up"
            else:
                if action.function_name == "pyautogui.hscroll":
                    actions[i].parameters["direction"] = "right"
                else:
                    actions[i].parameters["direction"] = "down"
            del actions[i].parameters["arg_0"]
            actions[i].function_name = "scroll"
            actions[i].parameters["amount"] = int(abs(arg_value * 100))

        elif action.function_name == "pyautogui.dragTo":
            actions[i].function_name = "drag"
            change_argument_name(actions[i])

        else:
            ValueError("Error FonctionCall Formatting")

        actions[i].original_string = actions[i].to_string()

    return actions

if __name__ == "__main__":
    from function_parser import FunctionCall

    # Example actions for all function types
    actions = [
        # MOBILE ACTIONS
        FunctionCall("mobile.home", {}, "mobile.home()"),
        FunctionCall("mobile.open_app", {"app_name": "drupe"}, "mobile.open_app(app_name='drupe')"),
        FunctionCall("mobile.swipe", {"from_coord": [0.581, 0.898], "to_coord": [0.601, 0.518]}, "mobile.swipe(from_coord=[0.581,0.898],to_coord=[0.601,0.518])"),
        FunctionCall("mobile.back", {}, "mobile.back()"),
        FunctionCall("mobile.long_press", {"x": 0.799, "y": 0.911}, "mobile.long_press(x=0.799, y=0.911)"),
        FunctionCall("mobile.terminate", {"status": "success"}, "mobile.terminate(status='success')"),
        FunctionCall("answer", {"arg_0": "text"}, "answer('text')"),
        FunctionCall("mobile.wait", {"seconds": 3}, "mobile.wait(seconds=3)"),
        # OS ACTIONS
        FunctionCall("pyautogui.hscroll", {"page": -0.1}, "pyautogui.hscroll(page=-0.1)"),
        FunctionCall("pyautogui.scroll", {"page": 0.13}, "pyautogui.scroll(page=0.13)"),
        FunctionCall("pyautogui.click", {"x": 0.8102, "y": 0.9463}, "pyautogui.click(x=0.8102, y=0.9463)"),
        FunctionCall("pyautogui.doubleClick", {}, "pyautogui.doubleClick()"),
        FunctionCall("pyautogui.hotkey", {"keys": ["ctrl", "c"]}, "pyautogui.hotkey(keys=['ctrl','c'])"),
        FunctionCall("pyautogui.press", {"keys": "enter"}, "pyautogui.press(keys='enter')"),
        FunctionCall("pyautogui.moveTo", {"x": 0.04, "y": 0.405}, "pyautogui.moveTo(x=0.04, y=0.405)"),
        FunctionCall("pyautogui.write", {"message": "bread buns"}, "pyautogui.write(message='bread buns')"),
        FunctionCall("pyautogui.dragTo", {"from_coord": [0.87, 0.423], "to_coord": [0.8102, 0.9463]}, "pyautogui.dragTo(from_coord=[0.87, 0.423], to_coord=[0.8102, 0.9463])"),
    ]
    resolution = (1080, 1920)
    print("Before conversion:")
    for action in actions:
        print(action)
    print("\nAfter conversion:")
    converted = action_conversion(actions, resolution)
    for action in converted:
        print(action)
