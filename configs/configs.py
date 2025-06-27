# Databricks notebook source
#import yaml
#import os
#from pathlib import Path
#import re
#
#def load_config(config_path):
#    with open(config_path, 'r') as file:
#        return yaml.safe_load(file)
#    
#def functions_path(file_path):
#    minus_pattern = r"(\/)\w+(\/)\w+$"
#    match = re.search(minus_pattern, file_path)
#    minus_str = match.group()
#    function_str = f"/Workspace{file_path.replace(minus_str,'')}"
#    return function_str
#
## Get the current working directory (where the notebook is located)
#current_notebook_path = Path(os.getcwd())
#
#global_config_path = str(current_notebook_path) + "/config_files/global_configs.yaml"
#global_config_path = global_config_path.replace("notebooks", "configs")
#
#data_prep_config_path = str(current_notebook_path) + "/config_files/configs.yaml"
#data_prep_config_path = data_prep_config_path.replace("notebooks", "configs")
#
#feature_config_path = str(current_notebook_path) + "/config_files/feature_engineering.yaml"
#feature_config_path = feature_config_path.replace("notebooks", "configs")
#
#training_config_path = str(current_notebook_path) + "/config_files/training.yaml"
#training_config_path = training_config_path.replace("notebooks", "configs")
#
#scoring_config_path = str(current_notebook_path) + "/config_files/scoring.yaml"
#scoring_config_path = scoring_config_path.replace("notebooks", "configs")
#
#serving_config_path = str(current_notebook_path) + "/config_files/serving.yaml"
#serving_config_path = serving_config_path.replace("notebooks", "configs")
#
#new_scoring_config_path = str(current_notebook_path) + "/config_files/scoring_new_pcd.yaml"
#new_scoring_config_path = new_scoring_config_path.replace("notebooks", "configs")
#
#def set_globals_from_config(config, parent_key=''):
#    """
#    Dynamically sets global variables for each leaf node in the configuration dictionary.
#    The variable names are created by concatenating the keys leading to each value.
#    """
#    for key, value in config.items():
#        # Construct the variable name based on the key hierarchy
#        var_name = f"{parent_key}_{key}" if parent_key else key
#        
#        if isinstance(value, dict):
#            set_globals_from_config(value, var_name)
#        else:
#            globals()[var_name] = value
#
#def extract_column_transformation_lists(path):
#    CONFIG_PATH = str(current_notebook_path) + path
#    CONFIG_PATH = CONFIG_PATH.replace("notebooks", "configs")
#    config = load_config(CONFIG_PATH)
#    set_globals_from_config(config)

# COMMAND ----------

import yaml
import os
from pathlib import Path
import re

def load_config(config_path):
    """Loads a YAML configuration file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
    
def functions_path(file_path):
    """
    Helper function to manipulate file paths.
    Assumes a specific directory structure and reconstructs a path.
    Example: /path/to/project/notebooks/feature_set/main.ipynb -> /Workspace/project/notebooks/feature_set
    More accurately, if file_path is like ".../levelA/levelB/levelC",
    it seems to aim to remove the last two levels and prepend "/Workspace".
    e.g. /foo/bar/baz/qux.txt -> /Workspace/foo/bar (if pattern matches /baz/qux.txt)
    Let's test the pattern: r"(\/)\w+(\/)\w+$"
    If file_path = "/part1/part2/file.py"
    match = re.search(r"(\/)\w+(\/)\w+$", "/part1/part2/file.py") -> matches "/part2/file.py"
    minus_str = "/part2/file.py"
    function_str = f"/Workspace{file_path.replace(minus_str, '')}" -> "/Workspace/part1"
    This seems to be the intended logic.
    """
    # Pattern to find the last two path components (e.g., /folder/file.yaml)
    minus_pattern = r"(\/)\w+(\/)\w+$" 
    match = re.search(minus_pattern, file_path)
    if match:
        minus_str = match.group(0) # Use group(0) for the whole match
        # Construct the new path by removing the matched part and prepending /Workspace
        # Ensure that file_path.replace only replaces the matched suffix
        # For example, if file_path is /a/b/c/d and minus_str is /c/d, result is /a/b
        # Then function_str is /Workspace/a/b
        base_path = file_path[:-len(minus_str)] if file_path.endswith(minus_str) else file_path.replace(minus_str, '')
        function_str = f"/Workspace{base_path}"
    else:
        # Handle cases where the pattern doesn't match, perhaps return original or modified path
        # For safety, let's return a clearly modified path or raise an error
        # This part depends on the expected structure of file_path
        # For now, returning a path that indicates no match or a default
        print(f"Warning: functions_path pattern did not match for {file_path}")
        function_str = f"/Workspace{file_path}" # Or handle as an error
    return function_str

# Get the current working directory (where the notebook is located)
# Path.cwd() is generally more robust for getting the current working directory.
try:
    # This assumes the script is run from a notebook environment where os.getcwd() is meaningful
    # For broader compatibility, consider passing base paths as arguments
    current_notebook_path_str = os.getcwd()
    current_notebook_path = Path(current_notebook_path_str)
except Exception as e:
    print(f"Error getting current working directory: {e}")
    print("Please ensure this script is run in an environment where os.getcwd() is valid or adjust path logic.")
    # Fallback or exit if necessary
    current_notebook_path = Path(".") # Default to current dir if getcwd fails in some contexts

# --- Configuration File Path Setup ---
# This logic assumes a directory structure like:
# project_root/
#  ├── configs/
#  │   └── config_files/
#  │       ├── global_configs.yaml
#  │       ├── configs.yaml (data_prep)
#  │       ├── feature_engineering.yaml
#  │       ├── training.yaml
#  │       ├── scoring.yaml
#  │       ├── serving.yaml
#  │       └── scoring_new_pcd.yaml
#  └── notebooks/  <- (where the notebook running this script is)

def get_config_file_path(notebook_base_path, relative_config_path):
    """
    Constructs the absolute path to a config file.
    It expects notebook_base_path to be a Path object.
    It replaces 'notebooks' segment in the path with 'configs'.
    Example:
        notebook_base_path = /path/to/project/notebooks
        relative_config_path = /config_files/global_configs.yaml
        Result: /path/to/project/configs/config_files/global_configs.yaml
    """
    # This assumes current_notebook_path_str is like ".../project_root/notebooks"
    # and we want to change it to ".../project_root/configs"
    # A more robust way might be to find a common project root.
    # For now, sticking to the user's original logic of string replacement.
    
    # Convert Path object to string for replacement, then back to Path if needed,
    # or operate with strings.
    base_path_str = str(notebook_base_path)
    
    # Check if "notebooks" is in the path to avoid errors if it's not.
    if "notebooks" in base_path_str:
        # Construct the path up to the 'notebooks' part, then switch to 'configs'
        # This is safer than a blind replace if "notebooks" could appear elsewhere.
        # Example: /home/user/my_notebooks_project/notebooks -> /home/user/my_notebooks_project/configs
        parts = list(notebook_base_path.parts)
        try:
            notebooks_index = parts.index("notebooks")
            # Construct path to project root (directory containing 'notebooks' and 'configs')
            project_root_parts = parts[:notebooks_index]
            config_dir_base_path = Path(*project_root_parts) / "configs"
        except ValueError:
            # 'notebooks' not found as a separate part, fall back to string replace or error
            print(f"Warning: 'notebooks' directory not found as expected in path {base_path_str}. Using string replacement.")
            config_dir_base_path_str = base_path_str.replace("notebooks", "configs", 1) # Replace only the first occurrence
            config_dir_base_path = Path(config_dir_base_path_str)
    else:
        # If "notebooks" isn't in the path, perhaps the script is already in a config-relative location
        # or the structure is different. For now, use the path as is for the base.
        print(f"Warning: 'notebooks' not found in path {base_path_str}. Assuming current path is relative to configs or adjust logic.")
        config_dir_base_path = notebook_base_path

    # Join with the relative config file path (stripping leading '/' if present in relative_config_path)
    # relative_config_path usually starts with /config_files/ so we might need to handle the join carefully
    if relative_config_path.startswith("/"):
        effective_relative_path = relative_config_path[1:]
    else:
        effective_relative_path = relative_config_path
        
    return str(config_dir_base_path / effective_relative_path)

# Define relative paths for config files
GLOBAL_CONFIG_REL_PATH = "/config_files/global_configs.yaml"
DATA_PREP_CONFIG_REL_PATH = "/config_files/configs.yaml"
FEATURE_CONFIG_REL_PATH = "/config_files/feature_engineering.yaml"
TRAINING_CONFIG_REL_PATH = "/config_files/training.yaml"
SCORING_CONFIG_REL_PATH = "/config_files/scoring.yaml"
SERVING_CONFIG_REL_PATH = "/config_files/serving.yaml"
NEW_SCORING_CONFIG_REL_PATH = "/config_files/scoring_new_pcd.yaml"

# Construct full paths
global_config_path = get_config_file_path(current_notebook_path, GLOBAL_CONFIG_REL_PATH)
data_prep_config_path = get_config_file_path(current_notebook_path, DATA_PREP_CONFIG_REL_PATH)
feature_config_path = get_config_file_path(current_notebook_path, FEATURE_CONFIG_REL_PATH)
training_config_path = get_config_file_path(current_notebook_path, TRAINING_CONFIG_REL_PATH)
scoring_config_path = get_config_file_path(current_notebook_path, SCORING_CONFIG_REL_PATH)
serving_config_path = get_config_file_path(current_notebook_path, SERVING_CONFIG_REL_PATH)
new_scoring_config_path = get_config_file_path(current_notebook_path, NEW_SCORING_CONFIG_REL_PATH)

# --- Main Logic ---
def set_globals_from_config(config, parent_key=''):
    """
    Dynamically sets global variables for each node in the configuration dictionary.
    The variable names are created by concatenating the keys leading to each value.
    MODIFIED: This version now sets a global variable for the dictionary itself
    before recursing into it.
    """
    for key, value in config.items():
        # Construct the variable name based on the key hierarchy
        # Sanitize key to ensure it's a valid Python variable name if necessary
        # For simplicity, assuming keys are already valid or simple strings.
        # If keys can have spaces or special chars, they would need sanitization.
        safe_key = re.sub(r'\W|^(?=\d)', '_', key) # Basic sanitization: replace non-alphanum with _, prefix numbers with _

        var_name = f"{parent_key}_{safe_key}" if parent_key else safe_key
        
        # Set the global variable for the current key-value pair.
        # If 'value' is a dictionary (e.g., rename_map), 
        # this line will make 'rename_map' (or 'parent_rename_map') a global dict.
        globals()[var_name] = value
        
        # If the value is a dictionary, also recurse to set its children as globals,
        # prefixed with the current var_name.
        if isinstance(value, dict):
            set_globals_from_config(value, var_name)
        # No 'else' needed here because globals()[var_name] = value handles leaf nodes too.

def extract_column_transformation_lists(relative_config_file_path):
    """
    Loads a specific config file and sets global variables from its contents.
    'relative_config_file_path' should be like "/config_files/training.yaml".
    """
    # Construct the full path to the config file
    # current_notebook_path is the Path object for the directory where the notebook (or script) is.
    # The get_config_file_path function handles the "notebooks" to "configs" replacement.
    CONFIG_PATH = get_config_file_path(current_notebook_path, relative_config_file_path)
    
    print(f"Loading configuration from: {CONFIG_PATH}")
    try:
        config = load_config(CONFIG_PATH)
        set_globals_from_config(config)
        print(f"Successfully loaded and set globals from {CONFIG_PATH}")
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {CONFIG_PATH}")
    except Exception as e:
        print(f"Error processing configuration file {CONFIG_PATH}: {e}")