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
