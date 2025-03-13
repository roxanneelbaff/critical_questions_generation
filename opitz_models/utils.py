import os
from datetime import datetime
from json_repair import repair_json

def extract_json_from_string(json_string: str, allow_lists: bool = False) -> dict:
    j = repair_json(json_string, return_objects=True)

    # If multiple distinct JSON objects are returned as a list of dictionaries, merge them into a single dictionary
    if not allow_lists:
        if isinstance(j, list) and all(isinstance(item, dict) for item in j):
            merged_json = {}
            for obj in j:
                merged_json.update(obj)
            j = merged_json
    return j

# Printing colored text to console
class bcolors:
    BOLD = "\033[1m"
    CYAN = "\033[96m"
    ORANGE = "\033[38;2;255;165;0m"
    PINK = "\033[38;2;255;105;180m"
    ENDC = "\033[0m"

def print_text_orange(text):
    print(bcolors.ORANGE + str(text) + bcolors.ENDC)
def print_text_cyan(text):
    print(bcolors.CYAN + str(text) + bcolors.ENDC)
def print_text_pink(text):
    print(bcolors.PINK + str(text) + bcolors.ENDC)
def print_text_bold(text):
    print(bcolors.BOLD + str(text) + bcolors.ENDC)

llm_calls = 0
def increment_llm_count():
    global llm_calls
    llm_calls += 1

def get_llm_calls():
    global llm_calls
    return llm_calls

def format_duration(start_time, end_time):
    duration = end_time - start_time
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    seconds = int(duration % 60)
    return f"{hours:02}h {minutes:02}m {seconds:02}s"

def log_message(file_path, message):
    # Get the current date and time in the desired format
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Format the log entry with date, time, and message
    log_entry = f"{current_time} - {message}\n"
    
    # Open the existing log file in append mode and write the log entry
    with open(file_path, "a") as log_file:
        log_file.write(log_entry)