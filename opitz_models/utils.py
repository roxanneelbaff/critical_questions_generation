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
    ENDC = "\033[0m"

def print_text_orange(text):
    print(bcolors.ORANGE + str(text) + bcolors.ENDC)
def print_text_cyan(text):
    print(bcolors.CYAN + str(text) + bcolors.ENDC)
def print_text_bold(text):
    print(bcolors.BOLD + str(text) + bcolors.ENDC)
