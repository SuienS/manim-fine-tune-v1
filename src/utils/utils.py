import re, os
import typing

def extract_manim_code_from_llm_response(response: str, mutliple_code_blocks: bool = False, select_index: int = -1) -> typing.Union[str, list[str]]:
    """
    Extracts the Manim code from the response string.

    Args:
        response (str): The response string containing the Manim code.
        mutliple_code_blocks (bool): If True, returns a list of code blocks. If False, returns a single code block.
        select_index (int): The index of the code block to return if mutliple_code_blocks is False. Preferred to be 0 or -1.

    Returns:
        str or list[str]: The extracted Manim code. If mutliple_code_blocks is True, returns a list of code blocks.
    """
    # Remove  '`<CODE>`' and '`</CODE>`' from the response
    response = re.sub(r"`<CODE>`|`</CODE>`", "", response)

    # Regular expression to match the code block
    code_block_pattern = r"<CODE>(.*?)</CODE>"

    # Python code block pattern
    python_code_block_pattern = r"```python(.*?)```"
    
    # Find all matches of the pattern in the response
    matches = re.findall(code_block_pattern, response, re.DOTALL)

    # If no matches found, try to find Python code block pattern
    if not matches:
        matches = re.findall(python_code_block_pattern, response, re.DOTALL)
    
    # If mutliple_code_blocks is True, return them as a list
    if mutliple_code_blocks:
        return [re.sub(r'<CODE>|</CODE>|```python|```', '', match).strip() for match in matches]
    
    # If mutliple_code_blocks is False, return the first match
    return re.sub(r'<CODE>|</CODE>|```python|```', '', matches[select_index]).strip() if matches else ""

def calculate_mean_from_dict_list(dict_list: list, key: str) -> float:
    """
    Calculate the mean of a specific key from a list of dictionaries.
    
    Args:
        dict_list (list): A list of dictionaries.
        key (str): The key to calculate the mean for.
    
    Returns:
        float: The mean value.
    """
    return sum(d[key] for d in dict_list) / len(dict_list)

def set_custom_cache_path():
    """
    Set the custom cache path for Hugging Face models.
    """
    from dotenv import load_dotenv 
    # loading variables from .env file
    load_dotenv()
    if os.getenv('CACHE_PATH') is not None:
        # Set the cache path of librarys to the one in .env file
        print("Setting cache path to: " + os.getenv('CACHE_PATH'))
        os.environ["HF_HOME"] = os.getenv('CACHE_PATH') + '/transformers'
        os.environ["HF_DATASETS_CACHE"] = os.getenv('CACHE_PATH') + '/datasets'
        os.environ["TORCH_HOME"] = os.getenv('CACHE_PATH') + '/torch'
        os.environ["TFHUB_CACHE_DIR"] = os.getenv('CACHE_PATH') + '/tfhub'
