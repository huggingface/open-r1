#!/usr/bin/env python3
"""
Function parser for extracting function names, parameter names, and values from string function calls.
Supports both mobile and pyautogui function patterns.
"""

import re
from typing import Dict, List, Tuple, Any, Union
from dataclasses import dataclass
from collections import OrderedDict

@dataclass
class FunctionCall:
    """Represents a parsed function call with its parameters."""
    function_name: str
    parameters: Dict[str, Any]
    original_string: str
    description: str = ""
    
    def to_string(self) -> str:
        """
        Reconstruct the function call string from the parsed data.
        
        Returns:
            String representation of the function call
            
        Examples:
            >>> call = FunctionCall("mobile.wait", {"seconds": 3}, "mobile.wait(seconds=3)")
            >>> call.to_string()
            "mobile.wait(seconds=3)"
            
            >>> call = FunctionCall("function", {"arg_0": 1, "arg_1": 2, "x": 0.5}, "function(1, 2, x=0.5)")
            >>> call.to_string()
            "function(1, 2, x=0.5)"
        """
        if not self.parameters:
            return f"{self.function_name}()"
        
        # Separate positional and named arguments
        positional_args = []
        named_args = []
        
        for name, value in self.parameters.items():
            if name.startswith("arg_"):
                # Positional argument
                positional_args.append((int(name.split("_")[1]), value))
            else:
                # kwargs
                named_args.append((name, value))
        
        # Sort positional arguments by index
        positional_args.sort(key=lambda x: x[0])
        
        # Build parameter string
        param_parts = []
        
        # Add positional arguments
        for _, value in positional_args:
            param_parts.append(self._value_to_string(value))
        
        # Add named arguments
        for name, value in named_args:
            param_parts.append(f"{name}={self._value_to_string(value)}")
        
        return f"{self.function_name}({', '.join(param_parts)})"
    
    def _value_to_string(self, value: Any) -> str:
        """
        Convert a value to its string representation for function calls.
        
        Args:
            value: The value to convert
            
        Returns:
            String representation of the value
        """
        if isinstance(value, str):
            # Quote strings
            return f"'{value}'"
        elif isinstance(value, (list, tuple)):
            # Convert lists/tuples to string representation
            items = [self._value_to_string(item) for item in value]
            return f"[{', '.join(items)}]"
        elif isinstance(value, dict):
            # Convert dictionaries to string representation
            items = [f"'{k}': {self._value_to_string(v)}" for k, v in value.items()]
            return f"{{{', '.join(items)}}}"
        elif isinstance(value, bool):
            # Convert booleans to lowercase
            return str(value).lower()
        elif value is None:
            return "None"
        else:
            # Numbers and other types
            return str(value)


def parse_function_call(function_string: str, pattern_to_match: list[str] = []) -> List[FunctionCall]:
    """
    Parse a function call string and extract all function calls found.
    
    Args:
        function_string: String representation of function calls
        
    Returns:
        List of FunctionCall objects with parsed information
        
    Examples:
        >>> parse_function_call("mobile.wait(seconds=3)")
        [FunctionCall(function_name='wait', parameters={'seconds': 3}, ...)]
        
        >>> parse_function_call("mobile. wait(seconds=3)")
        [FunctionCall(function_name='wait', parameters={'seconds': 3}, ...)]
        
        >>> parse_function_call("mobile.wait(seconds=3) mobile.home()")
        [FunctionCall(function_name='wait', parameters={'seconds': 3}, ...), FunctionCall(function_name='home', parameters={}, ...)]
    """
    # Remove any leading/trailing whitespace
    function_string = function_string.strip()
    
    # Pattern to match function calls with parameters
    # Matches: function_name(param1=value1, param2=value2, ...)
    # Can have any characters before the function call, extracts just the function name
    pattern = r'.*?([a-zA-Z_][a-zA-Z0-9_.]*)\(([^)]*)\)'
    
    matches = re.findall(pattern, function_string)
    if not matches:
        # No valid function calls found in: {function_string}
        return []
    
    results = []
    for match in matches:
        function_name = match[0]
        params_string = match[1]
        
        if pattern_to_match and all(pattern not in function_name for pattern in pattern_to_match):
            continue
        
        # Parse parameters
        parameters = parse_parameters(params_string)
        
        # Create the original string for this specific function call
        original_string = f"{function_name}({params_string})"
        
        results.append(FunctionCall(
            function_name=function_name,
            parameters=parameters,
            original_string=original_string
        ))
    
    return results


def parse_parameters(params_string: str) -> Dict[str, Any]:
    """
    Parse parameter string and extract parameter names and values.
    
    Args:
        params_string: String containing parameters (e.g., "x=0.5, y=0.6, text='hello'")
        
    Returns:
        Dictionary mapping parameter names to their values
        
    Examples:
        >>> parse_parameters("x=0.5, y=0.6")
        {'x': 0.5, 'y': 0.6}
        
        >>> parse_parameters("app_name='drupe'")
        {'app_name': 'drupe'}
        
        >>> parse_parameters("'text'")
        {'arg_0': 'text'}
        
        >>> parse_parameters("1, 3, 4")
        {'arg_0': 1, 'arg_1': 3, 'arg_2': 4}
        
        >>> parse_parameters("arg1, arg2, x=0.5")
        {'arg_0': 'arg1', 'arg_1': 'arg2', 'x': 0.5}
    """
    if not params_string.strip():
        return {}
    
    parameters = OrderedDict()
    
    # Split by commas, but be careful with commas inside quotes or brackets
    param_parts = split_parameters(params_string)
    
    positional_index = 0
    
    for part in param_parts:
        part = part.strip()
        if not part:
            continue
            
        # Parse individual parameter
        name, value = parse_single_parameter(part)
        
        # For positional arguments, use index-based naming
        if name.startswith("arg_"):
            name = f"arg_{positional_index}"
            positional_index += 1
        
        parameters[name] = value
    
    return parameters


def split_parameters(params_string: str) -> List[str]:
    """
    Split parameter string by commas, respecting quotes and brackets.
    
    Args:
        params_string: String containing parameters
        
    Returns:
        List of individual parameter strings
    """
    parts = []
    current_part = ""
    paren_count = 0
    bracket_count = 0
    brace_count = 0
    in_quotes = False
    quote_char = None
    
    for char in params_string:
        if char in ['"', "'"] and (not in_quotes or char == quote_char):
            if not in_quotes:
                in_quotes = True
                quote_char = char
            else:
                in_quotes = False
                quote_char = None
        elif not in_quotes:
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
            elif char == '[':
                bracket_count += 1
            elif char == ']':
                bracket_count -= 1
            elif char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
            elif char == ',' and paren_count == 0 and bracket_count == 0 and brace_count == 0:
                parts.append(current_part.strip())
                current_part = ""
                continue
        
        current_part += char
    
    if current_part.strip():
        parts.append(current_part.strip())
    
    return parts


def parse_single_parameter(param_string: str) -> Tuple[str, Any]:
    """
    Parse a single parameter string into name and value.
    
    Args:
        param_string: String like "x=0.5" or "app_name='drupe'" or just "value"
        
    Returns:
        Tuple of (parameter_name, parameter_value)
        
    Examples:
        >>> parse_single_parameter("x=0.5")
        ('x', 0.5)
        
        >>> parse_single_parameter("app_name='drupe'")
        ('app_name', 'drupe')
        
        >>> parse_single_parameter("'text'")
        ('arg_0', 'text')
        
        >>> parse_single_parameter("123")
        ('arg_0', 123)
        
        >>> parse_single_parameter("3")
        ('arg_0', 3)
    """
    # Pattern to match parameter name and value
    pattern = r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.+)$'
    
    match = re.match(pattern, param_string)
    if match:
        # Named parameter
        param_name = match.group(1)
        param_value_str = match.group(2).strip()
        param_value = parse_value(param_value_str)
        return param_name, param_value
    else:
        # Positional parameter - treat as unnamed argument
        param_value = parse_value(param_string)
        return "arg_0", param_value


def parse_value(value_string: str) -> Any:
    """
    Parse a value string into appropriate Python type.
    
    Args:
        value_string: String representation of a value
        
    Returns:
        Parsed value (int, float, str, list, etc.)
        
    Examples:
        >>> parse_value("3")
        3
        
        >>> parse_value("3.14")
        3.14
        
        >>> parse_value("'hello'")
        'hello'
        
        >>> parse_value("[0.581, 0.898]")
        [0.581, 0.898]
    """
    value_string = value_string.strip()
    
    # String values (quoted)
    if (value_string.startswith("'") and value_string.endswith("'")) or \
       (value_string.startswith('"') and value_string.endswith('"')):
        return value_string[1:-1]
    
    # List values
    if value_string.startswith('[') and value_string.endswith(']'):
        return parse_list(value_string)
    
    # Dictionary values
    if value_string.startswith('{') and value_string.endswith('}'):
        return parse_dict(value_string)
    
    # Boolean values
    if value_string.lower() in ['true', 'false']:
        return value_string.lower() == 'true'
    
    # None value
    if value_string.lower() == 'none':
        return None
    
    # Numeric values
    try:
        # Try integer first
        if '.' not in value_string:
            return int(value_string)
        else:
            return float(value_string)
    except ValueError:
        # If it's not a number, return as string (remove quotes if present)
        if value_string.startswith("'") and value_string.endswith("'"):
            return value_string[1:-1]
        elif value_string.startswith('"') and value_string.endswith('"'):
            return value_string[1:-1]
        else:
            return value_string


def parse_list(list_string: str) -> List[Any]:
    """
    Parse a list string into a Python list.
    
    Args:
        list_string: String like "[0.581, 0.898]"
        
    Returns:
        List of parsed values
        
    Examples:
        >>> parse_list("[0.581, 0.898]")
        [0.581, 0.898]
    """
    # Remove outer brackets
    content = list_string[1:-1].strip()
    if not content:
        return []
    
    # Split by commas, respecting nested structures
    parts = split_parameters(content)
    
    return [parse_value(part.strip()) for part in parts]


def parse_dict(dict_string: str) -> Dict[str, Any]:
    """
    Parse a dictionary string into a Python dict.
    
    Args:
        dict_string: String like "{'key': 'value'}"
        
    Returns:
        Dictionary of parsed key-value pairs
    """
    # Remove outer braces
    content = dict_string[1:-1].strip()
    if not content:
        return {}
    
    # Split by commas, respecting nested structures
    parts = split_parameters(content)
    
    result = {}
    for part in parts:
        part = part.strip()
        if ':' in part:
            key_str, value_str = part.split(':', 1)
            key = parse_value(key_str.strip())
            value = parse_value(value_str.strip())
            result[key] = value
    
    return result


def parse_multiple_functions(function_strings: List[str]) -> List[FunctionCall]:
    """
    Parse multiple function call strings.
    
    Args:
        function_strings: List of function call strings
        
    Returns:
        List of FunctionCall objects
    """
    results = []
    for func_str in function_strings:
        try:
            result_list = parse_function_call(func_str)
            results.extend(result_list)
        except Exception as e:
            print(f"Warning: Could not parse function call '{func_str}': {e}")
            continue
    
    return results


def extract_function_calls_from_text(text: str) -> List[FunctionCall]:
    """
    Extract and parse function calls from a text block.
    
    Args:
        text: Text containing function calls
        
    Returns:
        List of FunctionCall objects
    """
    # Pattern to find function calls in text
    # Matches: function_name(param1=value1, param2=value2)
    pattern = r'[a-zA-Z_][a-zA-Z0-9_.]*\([^)]*\)'
    
    matches = re.findall(pattern, text)
    return parse_multiple_functions(matches)


# Example usage and testing
if __name__ == "__main__":
    test_cases = [
        "mobile.home()",
        "mobile.open_app(app_name='drupe')",
        "mobile.swipe(from_coord=[0.581, 0.898], to_coord=[0.601, 0.518])",
        "mobile.back()",
        "mobile.long_press(x=0.799, y=0.911)",
        "mobile.terminate(status='success')",
        "answer('text')",
        "pyautogui.hscroll(page=-0.1)",
        "pyautogui.scroll(page=-0.1)",
        "pyautogui.scroll(0.13)",
        "pyautogui.click(x=0.8102, y=0.9463)",
        "pyautogui.hotkey(keys=['ctrl', 'c'])",
        "pyautogui.doubleClick()",
        "pyautogui.press(keys='enter')",
        "pyautogui.press(keys=['enter'])",
        "pyautogui.moveTo(x=0.04, y=0.405)",
        "pyautogui.write(message='bread buns')",
        "pyautogui.dragTo(x=0.8102, y=0.9463)",
        "mobile.wait(seconds=3)\nmobile.swipe(from_coord=[0.581, 0.898], to_coord=[0.601, 0.518])",
        # Additional test cases for multiple positional arguments
        "function(arg1, arg2, arg3)",
        "function('hello', 123, x=0.5)",
        "function(arg1, arg2, named_param='value')",
        "function(1, 2, 3, 4, 5)",
        "function('a', 'b', 'c', x=1, y=2)",
    ]
    
    print("Testing function parser:")
    print("=" * 50)
    
    for test_case in test_cases:
        try:
            results = parse_function_call(test_case)
            print(f"✓ {test_case}")
            for result in results:
                print(f"  Function: {result.function_name}")
                print(f"  Parameters: {result.parameters}")
            print()
        except Exception as e:
            print(f"✗ {test_case}")
            print(f"  Error: {e}")
            print()
    
    # Test extracting from text
    print("Testing text extraction:")
    print("=" * 50)
    
    sample_text = """
    mobile.wait(seconds=3)
    mobile.open_app(app_name='drupe')
    pyautogui.click(x=0.8102, y=0.9463)
    pyautogui.write(message='bread buns')
    """
    
    extracted = extract_function_calls_from_text(sample_text)
    for func_call in extracted:
        print(f"Found: {func_call.function_name} with params: {func_call.parameters}")
    
    # Test reconstruction
    print("\nTesting function call reconstruction:")
    print("=" * 50)
    
    reconstruction_tests = [
        "mobile.wait(seconds=3)",
        "mobile.home()",
        "mobile.open_app(app_name='drupe')",
        "mobile.swipe(from_coord=[0.581, 0.898], to_coord=[0.601, 0.518])",
        "answer('text')",
        "pyautogui.scroll(0.13)",
        "pyautogui.click(x=0.8102, y=0.9463)",
        "pyautogui.hotkey(keys=['ctrl', 'c'])",
        "function(1, 2, 3)",
        "function('hello', 123, x=0.5, y=0.8)",
        "function([1, 3], 'arg2', named_param='value')",
    ]
    
    for test_case in reconstruction_tests:
        parsed_list = parse_function_call(test_case)
        for parsed in parsed_list:
            reconstructed = parsed.to_string()
            print(f"Original:  {test_case}")
            print(f"Reconstructed: {reconstructed}")
            print(f"Match: {test_case == reconstructed}")
            assert test_case == reconstructed
            print() 