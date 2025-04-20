# --- START OF FILE idea_generator.py ---

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import random
import argparse
import json
import datetime
import subprocess
import re
# import glob # Not currently used, can be removed if not needed later
import logging
import time
import tempfile # Use tempfile for secure temporary file handling
from typing import List, Dict, Optional, Any, Tuple
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# === Configuration and Constants ===
MODELL_NAME = "microsoft/phi-4" # Or another suitable model
API_BASE_URL = "https://api.studio.nebius.com/v1/"
DEFAULT_PROPOSALS_FOLDER = "proposals" # Default folder for auto-saved proposals
DEFAULT_TARGET_WORDS_MIN = 5000       # Default minimum word count for generated ideas
DEFAULT_TARGET_WORDS_MAX = 10000      # Default maximum word count for generated ideas
STORY_GENERATOR_SCRIPT_NAME = "story_generator.py" # Name of the story generator script

# Retry Constants
DEFAULT_RETRY_DELAY_S: int = 10
MAX_RETRIES: int = 3
RETRY_BACKOFF_FACTOR: float = 1.5

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PromptSettingGenerator:
    """
    Generates random prompts and settings for short stories based on genres.
    Saves proposals and can optionally trigger the story generator script.
    """

    # Language configuration map (de/en keys)
    LANGUAGE_CONFIG: Dict[str, Dict[str, Any]] = {
        "de": {
            # --- German Prompts and UI Text ---
            "PROMPT_SYSTEM": """Du bist ein kreativer Ideengenerator für Geschichten. Deine Aufgabe ist es, Prompts und Settings für fesselnde Geschichten zu erstellen.
Genres/Themen: {genres_str}
Generiere {anzahl} verschiedene Ideen.
Für jede Idee:
1. Prompt: Detaillierte Prämisse/Grundidee (100-200 Wörter).
2. Setting: Passende atmosphärische Beschreibung (Ort, Zeit, Stimmung) (50-100 Wörter).
3. Titel: Kurzer, einprägsamer Titel.
4. Genre: Das Hauptgenre dieser Idee (aus den Vorgaben).
5. Wortanzahl: Eine zufällige Zahl zwischen {min_worte} und {max_worte}.

WICHTIG: Die generierten Titel, Prompts und Settings MÜSSEN auf Deutsch sein.

Formatiere deine Antwort NUR als valides JSON-Array:
[
    {{
        "titel": "...",
        "prompt": "...",
        "setting": "...",
        "genre": "...",
        "wortanzahl": ...
    }},
    ...
]
Sei kreativ und originell. Füge unerwartete Wendungen und interessante Konflikte ein. Gib NUR das JSON zurück, ohne einleitenden oder abschließenden Text.""",
            "USER_PROMPT": "Generiere {anzahl} kreative Geschichtsideen als JSON basierend auf Genres: {genres_str}.",
            "ERROR_MISSING_API_KEY": "API-Schlüssel ist erforderlich. Bitte NEBIUS_API_KEY setzen oder übergeben.",
            "ERROR_API_OVERLOAD": "API überlastet/Timeout. Warte {delay:.1f}s, versuche erneut ({retries}/{max_retries})...",
            "ERROR_API_CALL_FAILED": "Fehler während API-Aufruf: {error}",
            "ERROR_ALL_RETRIES_FAILED": "Alle API-Wiederholungen fehlgeschlagen.",
            "INFO_GENERATING": "Generiere {anzahl} Vorschläge basierend auf: {genres_str}...",
            "INFO_WAITING_API": "Warte auf API-Antwort...",
            "ERROR_JSON_PARSE": "Fehler beim Parsen des JSON: {error}",
            "INFO_TRY_REPAIR_JSON": "Versuche, teilweise JSON-Daten zu extrahieren...",
            "INFO_EXTRACTED_PARTIAL": "Erfolgreich {count} Vorschläge aus unvollständiger Antwort extrahiert!",
            "ERROR_NO_JSON_FOUND": "Konnte kein JSON in der Antwort finden.",
            "ERROR_EXTRACTION_FAILED": "Konnte keine Vorschläge aus der Antwort extrahieren.",
            "LABEL_TITLE": "Titel",
            "LABEL_GENRE": "Genre",
            "LABEL_WORDCOUNT": "Wortanzahl",
            "LABEL_PROMPT": "Prompt",
            "LABEL_SETTING": "Setting",
            "CMD_GENERATOR_SCRIPT": STORY_GENERATOR_SCRIPT_NAME, # Use constant
            "CMD_PYTHON_EXECUTABLE": sys.executable, # Use the current Python executable
            "CMD_SAVE_TEXT_ARG": "--save-text",
            "CMD_OUTPUT_DIR_ARG": "--output-dir",
            "CMD_LANG_MAP": {"de": "Deutsch", "en": "Englisch"}, # Map code to full name for story_generator
            "INFO_SAVED_PROPOSAL": "Vorschlag gespeichert als: {dateiname}",
            "WARN_INCOMPLETE_PROPOSAL": "Vorschlag {idx} unvollständig (fehlende Felder: {missing}). Überspringe Speichern.",
            "INFO_AUTO_SAVING_ALL": "Speichere alle Vorschläge automatisch im Ordner: {folder}",
            "INFO_PROPOSAL_SUMMARY": "Vorschlag {idx}: \"{titel}\" ({genre}, {wortanzahl} Wörter)",
            "PROMPT_SUMMARY": "Prompt: {text}...",
            "SETTING_SUMMARY": "Setting: {text}...",
            "PROMPT_CHOOSE_EXECUTE": "\nWelchen Vorschlag möchten Sie ausführen? (1-{max_idx}) oder 'q' zum Beenden: ",
            "PROMPT_INVALID_INPUT": "Bitte geben Sie eine gültige Zahl ein.",
            "PROMPT_INVALID_RANGE": "Bitte geben Sie eine Zahl zwischen 1 und {max_idx} ein.",
            "INFO_RANDOM_CHOICE": "\nZufällig ausgewählter Vorschlag zur Ausführung: {idx}",
            "INFO_EXECUTING": "\nFühre aus: \"{titel}\"",
            "INFO_RUNNING_GENERATION": "Führe Story-Generierung aus...",
            "ERROR_SAVING_PROPOSAL": "Fehler beim Speichern von Vorschlag {idx}: {error}"
        },
        "en": {
            # --- English Prompts and UI Text ---
            "PROMPT_SYSTEM": """You are an imaginative story idea generator. Your task is to create prompts and settings for captivating stories.
Genres/Themes: {genres_str}
Generate {anzahl} different ideas.
For each idea:
1. Prompt: Detailed premise/basic idea (100-200 words).
2. Setting: Appropriate atmospheric description (place, time, mood) (50-100 words).
3. Title: Short, memorable title.
4. Genre: The main genre of this idea (from the provided list).
5. Wordcount: A random number between {min_worte} and {max_worte}.

IMPORTANT: The generated titles, prompts, and settings MUST be in English.

Format your response ONLY as a valid JSON array:
[
    {{
        "title": "...",
        "prompt": "...",
        "setting": "...",
        "genre": "...",
        "wordcount": ...
    }},
    ...
]
Be creative and original. Add unexpected twists and interesting conflicts. Return ONLY the JSON, with no introductory or concluding text.""",
            "USER_PROMPT": "Generate {anzahl} creative story ideas as JSON based on genres: {genres_str}.",
            "ERROR_MISSING_API_KEY": "API key required. Please set NEBIUS_API_KEY or pass directly.",
            "ERROR_API_OVERLOAD": "API overloaded/timeout. Wait {delay:.1f}s, retry ({retries}/{max_retries})...",
            "ERROR_API_CALL_FAILED": "Error during API call: {error}",
            "ERROR_ALL_RETRIES_FAILED": "All API retries failed.",
            "INFO_GENERATING": "Generating {anzahl} proposals based on: {genres_str}...",
            "INFO_WAITING_API": "Waiting for API response...",
            "ERROR_JSON_PARSE": "Error parsing JSON: {error}",
            "INFO_TRY_REPAIR_JSON": "Attempting to extract partial JSON data...",
            "INFO_EXTRACTED_PARTIAL": "Successfully extracted {count} proposals from incomplete response!",
            "ERROR_NO_JSON_FOUND": "Could not find JSON in the response.",
            "ERROR_EXTRACTION_FAILED": "Could not extract proposals from the response.",
            "LABEL_TITLE": "Title",
            "LABEL_GENRE": "Genre",
            "LABEL_WORDCOUNT": "Wordcount",
            "LABEL_PROMPT": "Prompt",
            "LABEL_SETTING": "Setting",
            "CMD_GENERATOR_SCRIPT": STORY_GENERATOR_SCRIPT_NAME, # Use constant
            "CMD_PYTHON_EXECUTABLE": sys.executable, # Use the current Python executable
            "CMD_SAVE_TEXT_ARG": "--save-text",
            "CMD_OUTPUT_DIR_ARG": "--output-dir",
            "CMD_LANG_MAP": {"de": "Deutsch", "en": "Englisch"}, # Map code to full name for story_generator
            "INFO_SAVED_PROPOSAL": "Proposal saved as: {dateiname}",
            "WARN_INCOMPLETE_PROPOSAL": "Proposal {idx} incomplete (missing fields: {missing}). Skipping save.",
            "INFO_AUTO_SAVING_ALL": "Auto-saving all proposals to folder: {folder}",
            "INFO_PROPOSAL_SUMMARY": "Proposal {idx}: \"{titel}\" ({genre}, {wortanzahl} words)",
            "PROMPT_SUMMARY": "Prompt: {text}...",
            "SETTING_SUMMARY": "Setting: {text}...",
            "PROMPT_CHOOSE_EXECUTE": "\nWhich proposal do you want to execute? (1-{max_idx}) or 'q' to quit: ",
            "PROMPT_INVALID_INPUT": "Please enter a valid number.",
            "PROMPT_INVALID_RANGE": "Please enter a number between 1 and {max_idx}.",
            "INFO_RANDOM_CHOICE": "\nRandomly selected proposal for execution: {idx}",
            "INFO_EXECUTING": "\nExecuting: \"{titel}\"",
            "INFO_RUNNING_GENERATION": "Running story generation...",
            "ERROR_SAVING_PROPOSAL": "Error saving proposal {idx}: {error}"
        }
    }

    def __init__(self, api_key: Optional[str] = None,
                 proposals_folder: str = DEFAULT_PROPOSALS_FOLDER,
                 model: str = MODELL_NAME):
        """
        Initializes the PromptSettingGenerator.
        Args:
            api_key: Nebius API key (can be None, checks environment variable).
            proposals_folder: Directory to store generated proposal files.
            model: Name of the language model to use.
        """
        resolved_api_key = api_key or os.environ.get("NEBIUS_API_KEY")
        # Use 'de' for initialization errors first, fallback to 'en' if needed
        lang_conf_init = self.LANGUAGE_CONFIG.get("de", self.LANGUAGE_CONFIG.get("en"))
        if not resolved_api_key:
            # Raise error if API key is not found
            raise ValueError(lang_conf_init["ERROR_MISSING_API_KEY"])

        self.client = OpenAI(base_url=API_BASE_URL, api_key=resolved_api_key)
        self.proposals_folder = proposals_folder
        self.model_name = model
        # Ensure the default proposals folder exists when the generator is created
        try:
            os.makedirs(self.proposals_folder, exist_ok=True)
        except OSError as e:
            logging.warning(f"Could not create proposals folder '{self.proposals_folder}': {e}")


    def _get_lang_config(self, language_code: str) -> Dict[str, Any]:
        """Gets the configuration dictionary for the specified language code (e.g., 'de', 'en')."""
        config = self.LANGUAGE_CONFIG.get(language_code.lower())
        if not config:
            # Raise error if the language code is not supported
            supported_langs = list(self.LANGUAGE_CONFIG.keys())
            raise ValueError(f"Language code '{language_code}' not supported. Available codes: {supported_langs}")
        return config

    def retry_api_call(self, call_function, *args, **kwargs):
            """
            Executes an API call with automatic retries using exponential backoff for specific errors.
            """
            retries = 0
            # Use 'de' config for retry messages, fallback to 'en'
            lang_conf_retry = self.LANGUAGE_CONFIG.get("de", self.LANGUAGE_CONFIG.get("en"))
            while retries <= MAX_RETRIES:
                try:
                    # Attempt the API call
                    return call_function(*args, **kwargs)
                except Exception as e:
                    error_str = str(e).lower()
                    # Check if the error is likely a temporary server issue
                    is_retryable = ("overloaded" in error_str or "rate_limit" in error_str or
                                    "timeout" in error_str or "503" in error_str or "504" in error_str)

                    if is_retryable and retries < MAX_RETRIES:
                        retries += 1
                        # Calculate delay with exponential backoff and jitter
                        delay = (DEFAULT_RETRY_DELAY_S * (RETRY_BACKOFF_FACTOR ** (retries - 1)) +
                                random.uniform(0.1, 0.5)) # Add random jitter
                        logging.warning(lang_conf_retry["ERROR_API_OVERLOAD"].format(
                            delay=delay, retries=retries, max_retries=MAX_RETRIES
                        ))
                        time.sleep(delay) # Wait before retrying
                    else:
                        # If error is not retryable or max retries reached, log and re-raise
                        logging.error(lang_conf_retry["ERROR_API_CALL_FAILED"].format(error=str(e)))
                        raise # Re-raise the original exception
            # This should not be reached if the loop works correctly, but raise error just in case
            raise Exception(lang_conf_retry["ERROR_ALL_RETRIES_FAILED"])

    def _create_system_prompt(self, genres: List[str], count: int, language_code: str) -> str:
        """Creates the system prompt content for the API request."""
        lang_conf = self._get_lang_config(language_code)
        genres_str = ', '.join(genres) # Comma-separated list of genres
        # Format the prompt template with provided details
        return lang_conf["PROMPT_SYSTEM"].format(
            genres_str=genres_str,
            anzahl=count, # 'anzahl' is the key in the prompt template
            min_worte=DEFAULT_TARGET_WORDS_MIN, # 'min_worte' key
            max_worte=DEFAULT_TARGET_WORDS_MAX # 'max_worte' key
        )

    def _parse_json_response(self, response_text: str, language_code: str) -> List[Dict[str, Any]]:
        """
        Attempts to extract and parse the JSON array from the API response string.
        Handles potential markdown code blocks and tries to repair partial JSON.
        """
        lang_conf = self._get_lang_config(language_code)
        proposals = []
        json_str = None

        # Attempt 1: Use regex to find JSON within ```json ... ``` or just [...]
        # re.DOTALL makes '.' match newlines too. re.IGNORECASE for `json`.
        match = re.search(r'```json\s*(\[.*?\])\s*```|(\[.*?\])', response_text, re.DOTALL | re.IGNORECASE)

        if match:
            # Group 1 captures content inside ```json blocks, Group 2 captures raw [...]
            json_str = match.group(1) or match.group(2)

        if json_str:
            try:
                # Try parsing the extracted string directly
                proposals = json.loads(json_str)
                logging.debug("JSON successfully parsed using regex and json.loads.")
            except json.JSONDecodeError as e:
                # If direct parsing fails, log the error and attempt repair
                logging.warning(lang_conf["ERROR_JSON_PARSE"].format(error=e))
                logging.info(lang_conf["INFO_TRY_REPAIR_JSON"])
                # Use the repair function on the *original* response_text, as regex might have missed parts
                proposals = self._repair_incomplete_json(response_text, language_code)
        else:
            # Attempt 2: If regex didn't find anything promising, try repairing the whole response
            logging.warning(lang_conf["ERROR_NO_JSON_FOUND"])
            logging.info(lang_conf["INFO_TRY_REPAIR_JSON"])
            proposals = self._repair_incomplete_json(response_text, language_code)

        if not proposals:
             # If still no proposals after repair attempt
             logging.error(lang_conf["ERROR_EXTRACTION_FAILED"])
             logging.debug(f"Raw API response:\n{response_text}")
             return []

        # Post-processing: Standardize keys and validate required fields
        processed_proposals = []
        # Define required keys (using German keys as they are target in repair function)
        required_keys = ["titel", "prompt", "setting", "genre", "wortanzahl"]

        # Check if the result is a list (expected)
        if not isinstance(proposals, list):
             logging.error(f"Expected a list of proposals, but got type {type(proposals)}. Response: {response_text}")
             return []


        for proposal_item in proposals:
            # Ensure each item is a dictionary
            if isinstance(proposal_item, dict):
                # Standardize keys (map English keys like 'title'/'wordcount' if present)
                proposal_item["titel"] = proposal_item.pop("title", proposal_item.get("titel"))
                proposal_item["wortanzahl"] = proposal_item.pop("wordcount", proposal_item.get("wortanzahl"))

                # Provide default values for optional/missing fields
                if "wortanzahl" not in proposal_item or not isinstance(proposal_item.get("wortanzahl"), int):
                     proposal_item["wortanzahl"] = random.randint(DEFAULT_TARGET_WORDS_MIN, DEFAULT_TARGET_WORDS_MAX)
                if "genre" not in proposal_item or not proposal_item.get("genre"):
                    proposal_item["genre"] = "[Unknown]" if language_code == 'en' else "[Unbekannt]"

                # Check if all essential fields are present and non-empty
                missing_keys = [k for k in required_keys if k not in proposal_item or not proposal_item[k]]
                if not missing_keys:
                    processed_proposals.append(proposal_item)
                else:
                    # Log skipped incomplete proposals
                    title_for_log = proposal_item.get('titel', '[No Title]')
                    logging.warning(f"Skipping incomplete proposal '{title_for_log}'. Missing/empty keys: {', '.join(missing_keys)}")
            else:
                # Log if an item in the list is not a dictionary
                logging.warning(f"Skipping invalid item in JSON response (expected dict, got {type(proposal_item)}): {proposal_item}")

        return processed_proposals


    def generate_proposals(self, genres: List[str], count: int = 3, language_code: str = "de") -> List[Dict[str, Any]]:
        """
        Generates proposals for story prompts and settings via the API.
        Args:
            genres: A list of genres/themes to base the ideas on.
            count: The number of proposals to generate.
            language_code: The language code ('de' or 'en') for the proposals.
        Returns:
            A list of dictionaries, each representing a story proposal, or an empty list on failure.
        """
        lang_conf = self._get_lang_config(language_code)
        logging.info(lang_conf["INFO_GENERATING"].format(anzahl=count, genres_str=', '.join(genres)))

        # Create the prompts for the API call
        system_prompt = self._create_system_prompt(genres, count, language_code)
        user_prompt = lang_conf["USER_PROMPT"].format(anzahl=count, genres_str=', '.join(genres))

        # Generate a timestamp for potential temporary filenames
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Create a secure temporary file path
        temp_fd, temp_filename = tempfile.mkstemp(prefix=f"temp_proposals_{timestamp}_", suffix=".txt")
        os.close(temp_fd) # Close the file descriptor, we only need the path

        try:
            logging.info(lang_conf["INFO_WAITING_API"])
            # Determine if the model supports direct JSON output format
            # Check model name - this is a basic check, might need refinement
            use_json_format = "json" in self.model_name.lower() or "phi-4" in self.model_name.lower() # Example check
            response_format_arg = {"type": "json_object"} if use_json_format else None

            if use_json_format:
                 logging.debug("Requesting JSON object format from the model.")

            # Make the API call with retry logic
            response = self.retry_api_call(
                self.client.chat.completions.create,
                model=self.model_name,
                max_tokens=max(1500, count * 500), # Increase tokens based on proposal count
                temperature=0.9, # Higher temperature for creative ideas
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format=response_format_arg # Request JSON format if supported
            )

            # Extract the response content
            response_content = response.choices[0].message.content

            # If the model returns a structured JSON object directly (due to response_format)
            # it might already be parsed. We need the string representation for parsing/repair.
            if isinstance(response_content, (dict, list)):
                 response_text = json.dumps(response_content)
                 logging.debug("API returned structured JSON, converted back to string for parsing.")
            else:
                 response_text = str(response_content) # Treat as string

            # Save the raw response text to the temporary file (for debugging)
            try:
                with open(temp_filename, 'w', encoding='utf-8') as f:
                    f.write(response_text)
                logging.debug(f"Raw API response saved to temporary file: {temp_filename}")
            except Exception as e:
                 logging.warning(f"Could not save raw response to temporary file: {e}")

            # Parse the response text to extract proposals
            return self._parse_json_response(response_text, language_code)

        except Exception as e:
            # Handle potential errors during the API call or parsing
            logging.error(f"Critical error during proposal generation: {str(e)}")
            # Attempt to recover from the temporary file if it exists
            if os.path.exists(temp_filename):
                try:
                    with open(temp_filename, 'r', encoding='utf-8') as f:
                        partial_response = f.read()
                    # Only attempt repair if there's substantial content
                    if len(partial_response) > 50:
                        logging.info(f"Attempting recovery from partial response ({len(partial_response)} chars).")
                        return self._parse_json_response(partial_response, language_code)
                except Exception as e2:
                    logging.error(f"Error reading temporary file during recovery: {str(e2)}")
            return [] # Return empty list on failure
        finally:
            # Always clean up the temporary file
            if os.path.exists(temp_filename):
                try:
                    os.remove(temp_filename)
                    logging.debug(f"Temporary file deleted: {temp_filename}")
                except Exception as e:
                    logging.warning(f"Could not delete temporary file: {e}")

    def _repair_incomplete_json(self, partial_response: str, language_code: str) -> List[Dict[str, Any]]:
        """
        Attempts to extract valid JSON objects from potentially incomplete or malformed JSON strings.
        This is a heuristic approach and might not capture all valid data.
        """
        lang_conf = self._get_lang_config(language_code)
        proposals = []
        logging.debug(f"Attempting to repair JSON from: {partial_response[:200]}...") # Log start of text

        # Try to find the start of a JSON array '[' or object '{'
        json_array_start = partial_response.find('[')
        json_object_start = partial_response.find('{')

        start_index = -1
        if json_array_start != -1 and (json_object_start == -1 or json_array_start < json_object_start):
            start_index = json_array_start
        elif json_object_start != -1:
            start_index = json_object_start

        if start_index == -1:
            logging.debug("No '[' or '{' found in the partial response.")
            return [] # Cannot find start of JSON structure

        raw_content = partial_response[start_index:]

        # Iterate through the string trying to find complete '{...}' blocks
        cursor = 0
        while cursor < len(raw_content):
            obj_start = raw_content.find('{', cursor)
            if obj_start == -1:
                break # No more starting braces found

            nest_level = 0
            obj_end = -1
            in_string = False
            escaped = False

            # Scan from the object start to find the matching closing brace '}'
            for i in range(obj_start, len(raw_content)):
                char = raw_content[i]

                if in_string:
                    if char == '"' and not escaped:
                        in_string = False
                    # Handle escaped backslash correctly
                    elif char == '\\':
                        escaped = not escaped
                    else:
                        escaped = False
                else: # Not inside a string
                    if char == '"':
                        in_string = True
                        escaped = False
                    elif char == '{':
                        nest_level += 1
                    elif char == '}':
                        nest_level -= 1
                        if nest_level == 0: # Found the matching closing brace
                            obj_end = i
                            break # Stop scanning for this object
                    # Ignore other characters outside strings for nesting level

                # Reset escape flag if the current char is not a backslash
                # This handles cases like "\\", where the second '\' is escaped by the first.
                if char != '\\':
                     escaped = False


            if obj_end > obj_start:
                # Found a potential complete object string
                object_str = raw_content[obj_start : obj_end + 1]
                logging.debug(f"Potential JSON object found: {object_str[:100]}...")
                try:
                    # Try parsing this individual object string
                    proposal = json.loads(object_str)
                    # Validate if it's a dictionary (basic check)
                    if isinstance(proposal, dict):
                        # Add the valid proposal (further validation happens in _parse_json_response)
                        proposals.append(proposal)
                    else:
                         logging.debug(f"Parsed object is not a dictionary: {type(proposal)}")

                except json.JSONDecodeError as json_err:
                    # Log if parsing the extracted object fails
                    logging.debug(f"Could not parse extracted JSON object: {json_err} - Object string: {object_str[:100]}...")
                    pass # Ignore invalid fragments

                # Move cursor to continue searching after the found object
                cursor = obj_end + 1
            else:
                # No matching closing brace found for the current starting brace
                logging.debug(f"No matching '}}' found for '{{' starting at index {obj_start + start_index} in original string.")
                break # Stop searching

        if proposals:
             # Use the correct language config key here
             count_key = len(proposals)
             logging.info(lang_conf["INFO_EXTRACTED_PARTIAL"].format(count=count_key))

        return proposals


    def _safe_filename(self, title: str) -> str:
        """Creates a safe filename string from a title."""
        # Remove characters that are not alphanumeric, underscore, hyphen, or whitespace
        safe_title = re.sub(r'[^\w\s-]', '', title).strip()
        # Replace whitespace sequences with a single underscore
        safe_title = re.sub(r'\s+', '_', safe_title)
        # Truncate to a reasonable length
        return safe_title[:50]

    def save_proposal_as_txt(self, proposal: Dict[str, Any], language_code: str,
                             output_path: Optional[str] = None,
                             use_proposals_folder: bool = False) -> str:
        """
        Saves a single proposal dictionary as a formatted text file.

        Handles path logic: Saves to `output_path` if provided (treating it as a
        directory or file based on its structure), or to `self.proposals_folder`
        if `use_proposals_folder` is True, otherwise defaults to the current directory.

        Args:
            proposal: The proposal dictionary.
            language_code: 'de' or 'en'.
            output_path: Specific directory or file path for saving.
            use_proposals_folder: If True, forces saving into `self.proposals_folder`.

        Returns:
            The full path of the saved file.
        Raises:
            OSError: If directory creation fails.
            IOError: If file writing fails.
        """
        lang_conf = self._get_lang_config(language_code)
        # Create a safe base filename from the title
        safe_title = self._safe_filename(proposal.get("titel", "No_Title" if language_code == 'en' else "Ohne_Titel"))
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"proposal_{safe_title}_{timestamp}.txt" if language_code == 'en' else f"vorschlag_{safe_title}_{timestamp}.txt"

        # --- Determine the final target directory and filename ---
        target_dir = "." # Default to current directory
        final_filename = base_filename # Default filename in current dir

        if use_proposals_folder:
            # Force saving to the dedicated proposals folder
            target_dir = self.proposals_folder
            final_filename = os.path.join(target_dir, base_filename)
        elif output_path:
            # User provided a specific output path
            output_path = os.path.normpath(output_path) # Normalize path separators
            # Check if it looks like a directory (ends in separator, is an existing dir, or has no extension)
            if output_path.endswith(os.sep) or os.path.isdir(output_path) or not os.path.splitext(output_path)[1]:
                 # Treat as directory
                 target_dir = output_path
                 final_filename = os.path.join(target_dir, base_filename)
            else:
                 # Treat as a specific file path
                 final_filename = output_path
                 target_dir = os.path.dirname(final_filename)
                 # If dirname is empty (e.g., "myfile.txt"), use current directory
                 if not target_dir:
                     target_dir = "."

        # --- Ensure the target directory exists ---
        try:
            # Create directory if it doesn't exist (idempotent)
            # Handle case where target_dir might be empty if only a filename was given
            os.makedirs(target_dir if target_dir else ".", exist_ok=True)
        except OSError as e:
            logging.error(f"Error creating directory '{target_dir}': {e}")
            raise # Propagate the error

        # Adjust final_filename if only directory was specified originally
        if os.path.isdir(final_filename): # If final_filename resolved to a directory
             final_filename = os.path.join(final_filename, base_filename)


        # --- Prepare the file content ---
        content = f"{lang_conf['LABEL_TITLE']}: {proposal.get('titel', '[N/A]')}\n"
        content += f"{lang_conf['LABEL_GENRE']}: {proposal.get('genre', '[N/A]')}\n"
        # Use 'wortanzahl' key consistently for word count
        content += f"{lang_conf['LABEL_WORDCOUNT']}: {proposal.get('wortanzahl', '[N/A]')}\n\n"
        content += f"{lang_conf['LABEL_PROMPT']}:\n{proposal.get('prompt', '[N/A]')}\n\n"
        content += f"{lang_conf['LABEL_SETTING']}:\n{proposal.get('setting', '[N/A]')}\n\n"

        # --- Add example execution commands ---
        story_gen_script = lang_conf['CMD_GENERATOR_SCRIPT']
        python_exec = lang_conf['CMD_PYTHON_EXECUTABLE']
        # Map language code ('de'/'en') to the full language name expected by story_generator.py
        lang_arg_story_gen = lang_conf['CMD_LANG_MAP'].get(language_code, language_code)

        # Prepare arguments, ensuring proper quoting/escaping for command line
        # Using repr() is a safe way to get a string representation suitable for shell commands
        prompt_arg_repr = repr(str(proposal.get('prompt', '')))
        setting_arg_repr = repr(str(proposal.get('setting', '')))
        title_arg_repr = repr(str(proposal.get('titel', '')))
        wordcount_arg = str(proposal.get('wortanzahl', '')) # Should be a string number

        # Example command (adjust based on actual story_generator args)
        # NOTE: Using f-strings with repr() for safer argument construction
        command_base = (
            f"{python_exec} {story_gen_script} "
            f"--prompt {prompt_arg_repr} "
            f"--setting {setting_arg_repr} "
            f"--title {title_arg_repr} " # Use --title consistently
            f"--wordcount {wordcount_arg} " # Use --wordcount consistently
            f"--language {lang_arg_story_gen} " # Use --language consistently
            f"{lang_conf['CMD_SAVE_TEXT_ARG']}" # Add --save-text flag
            # Consider adding --output-dir argument here if needed
            # f" {lang_conf['CMD_OUTPUT_DIR_ARG']} {repr(target_dir)}" # Example
        )

        # Add comments for different shells
        win_cmd_example = f"REM Windows (cmd):\n{command_base}"
        unix_cmd_example = f"# Unix/Linux/MacOS (bash):\n{command_base}"

        content += "=== Example Execution Command (check quotes/escaping!) ===\n"
        content += win_cmd_example + "\n\n"
        content += unix_cmd_example + "\n"

        # --- Write the file ---
        try:
            with open(final_filename, 'w', encoding='utf-8') as f:
                f.write(content)
            logging.info(lang_conf["INFO_SAVED_PROPOSAL"].format(dateiname=final_filename))
            return final_filename # Return the path of the saved file
        except IOError as e:
            logging.error(f"Error writing file '{final_filename}': {e}")
            raise # Propagate the error


# === Main Execution Function ===
def main():
    parser = argparse.ArgumentParser(
        description="Random generator for short story prompts and settings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--genres", type=str, nargs="*",
                        help="Genres/themes for generation (multiple possible, see code for defaults)")
    parser.add_argument("--language", "--lang", type=str, default="de", choices=["de", "en"], # Added alias --lang
                        dest="language_code", # Store in 'language_code' variable
                        help="Language code for generated proposals (de/en)")
    parser.add_argument("--count", type=int, default=3, dest="proposal_count", # Store in 'proposal_count'
                        help="Number of proposals to generate")
    parser.add_argument("--api-key", type=str,
                        help="Nebius API key (alternatively use NEBIUS_API_KEY environment variable)")
    parser.add_argument("--model", type=str, default=MODELL_NAME,
                        help="Name of the LLM model to use")
    parser.add_argument("--select", action="store_true", dest="interactive_select", # Store in 'interactive_select'
                        help="Interactively select a proposal to execute")
    parser.add_argument("--execute", action="store_true", dest="execute_selected", # Store in 'execute_selected'
                        help=f"Execute selected (or random) proposal using '{STORY_GENERATOR_SCRIPT_NAME}'")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Directory for saving individual proposal text files (if not using --auto-save)")
    parser.add_argument("--proposals-folder", type=str, default=DEFAULT_PROPOSALS_FOLDER,
                        help="Folder for automatically saving all proposals (used with --auto-save)")
    # --ignore-existing is not implemented, removed for clarity. Add back if functionality is added.
    # parser.add_argument("--ignore-existing", action="store_true",
    #                     help="Placeholder: Currently does not ignore existing proposals")
    parser.add_argument("--auto-save", action="store_true",
                        help=f"Automatically save all generated proposals to the folder specified by --proposals-folder")
    parser.add_argument("--debug", action="store_true", help="Enable detailed debug logging")

    args = parser.parse_args()

    # --- Configure Logging Level ---
    log_level = logging.DEBUG if args.debug else logging.INFO
    # Force reconfiguration to apply the level correctly
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
    logging.debug("Debug mode enabled.")


    # --- Prepare Genres ---
    # Default list of genres if none are provided by the user
    default_genres = ["Sherlock Holmes", "Science Fiction", "Fantasy", "Horror", "Mystery", "Dark Fantasy", "Cyberpunk"]
    if not args.genres:
        args.genres = default_genres
        logging.info(f"No genres provided, using defaults: {', '.join(default_genres)}")

    # Randomly select 1-3 genres if many are provided, otherwise use all provided
    if len(args.genres) > 3:
        # Choose a random number k between 1 and 3 (inclusive)
        k = random.randint(1, 3)
        selected_genres = random.sample(args.genres, k=k)
        logging.info(f"Randomly selected {k} genres for generation: {', '.join(selected_genres)}")
    else:
        selected_genres = args.genres
        logging.info(f"Using provided genres: {', '.join(selected_genres)}")

    # --- Initialize Generator ---
    generator = None # Initialize to allow access in finally block if init fails
    try:
        generator = PromptSettingGenerator(
            api_key=args.api_key,
            proposals_folder=args.proposals_folder,
            model=args.model
        )
        # Get language config based on user input *after* generator is initialized
        lang_conf = generator._get_lang_config(args.language_code)

        # --- Generate Proposals ---
        proposals = generator.generate_proposals(
            selected_genres,
            args.proposal_count,
            args.language_code
        )

        if not proposals:
            # Log error and exit if no proposals could be generated
            logging.error("Failed to generate any proposals.")
            sys.exit(1)

        # --- Display Generated Proposals ---
        print("\n=== Generated Proposals ===")
        for i, proposal in enumerate(proposals, 1):
            # Use .get() for safe access to dictionary keys
            title = proposal.get("titel", "[No Title]")
            genre = proposal.get("genre", "[Unknown]")
            wordcount = proposal.get("wortanzahl", "[N/A]") # Use 'wortanzahl' key
            prompt_text = proposal.get("prompt", "")
            setting_text = proposal.get("setting", "")

            # Display summary using language-specific labels
            print(lang_conf["INFO_PROPOSAL_SUMMARY"].format(idx=i, titel=title, genre=genre, wortanzahl=wordcount))
            # Show truncated prompt/setting for brevity
            prompt_summary = prompt_text[:150] + "..." if len(prompt_text) > 150 else prompt_text
            setting_summary = setting_text[:100] + "..." if len(setting_text) > 100 else setting_text
            print(f"{lang_conf['LABEL_PROMPT']}: {prompt_summary}")
            print(f"{lang_conf['LABEL_SETTING']}: {setting_summary}")
            print("-" * 20)


        # --- Save Proposals ---
        saved_files: List[str] = []
        required_keys_for_saving = ["titel", "prompt", "setting", "genre", "wortanzahl"]

        if args.auto_save:
            # Save all proposals to the dedicated proposals folder
            logging.info(lang_conf["INFO_AUTO_SAVING_ALL"].format(folder=args.proposals_folder))
            for i, proposal in enumerate(proposals, 1):
                 # Check for missing essential keys before attempting to save
                 missing_keys = [k for k in required_keys_for_saving if k not in proposal or not proposal[k]]
                 if missing_keys:
                     logging.warning(lang_conf["WARN_INCOMPLETE_PROPOSAL"].format(idx=i, missing=', '.join(missing_keys)))
                     continue # Skip saving incomplete proposals
                 try:
                     # Call saving function, forcing use of proposals_folder
                     filename = generator.save_proposal_as_txt(proposal, args.language_code, use_proposals_folder=True)
                     saved_files.append(filename)
                 except Exception as e:
                     # Log errors during saving
                     logging.error(lang_conf["ERROR_SAVING_PROPOSAL"].format(idx=i, error=e))
        else:
             # Save proposals individually to the specified output directory (or current if '.')
             # This loop is redundant if --auto-save is not used, but kept for clarity
             # and potential future use cases where saving individual files without auto-save is desired.
             logging.info(f"Saving proposal details to: {args.output_dir}")
             for i, proposal in enumerate(proposals, 1):
                 missing_keys = [k for k in required_keys_for_saving if k not in proposal or not proposal[k]]
                 if missing_keys:
                     logging.warning(lang_conf["WARN_INCOMPLETE_PROPOSAL"].format(idx=i, missing=', '.join(missing_keys)))
                     continue
                 try:
                     # Pass the general output_dir for saving location
                     filename = generator.save_proposal_as_txt(proposal, args.language_code, output_path=args.output_dir)
                     saved_files.append(filename)
                 except Exception as e:
                      logging.error(lang_conf["ERROR_SAVING_PROPOSAL"].format(idx=i, error=e))


        # --- Interactive Selection / Execution ---
        selected_proposal_index: Optional[int] = None
        if args.execute_selected: # Check if execution is requested
            if args.interactive_select:
                # Prompt user for interactive selection
                while True:
                    try:
                        max_idx = len(proposals)
                        input_str = input(lang_conf["PROMPT_CHOOSE_EXECUTE"].format(max_idx=max_idx))
                        if input_str.lower() == 'q':
                            # User chose to quit
                            logging.info("Exiting.")
                            sys.exit(0)
                        # Convert input to 0-based index
                        selected_proposal_index = int(input_str) - 1
                        # Validate the selected index
                        if 0 <= selected_proposal_index < max_idx:
                            break # Valid selection
                        else:
                            print(lang_conf["PROMPT_INVALID_RANGE"].format(max_idx=max_idx))
                    except ValueError:
                        # Handle non-integer input
                        print(lang_conf["PROMPT_INVALID_INPUT"])
            else:
                # No interactive selection, choose randomly if proposals exist
                if proposals:
                    selected_proposal_index = random.randint(0, len(proposals) - 1)
                    logging.info(lang_conf["INFO_RANDOM_CHOICE"].format(idx=selected_proposal_index + 1))
                else:
                    # Should not happen if generation succeeded, but handle defensively
                    logging.warning("No proposals available to randomly select for execution.")


            # --- Execute the selected proposal ---
            if selected_proposal_index is not None:
                chosen_proposal = proposals[selected_proposal_index]
                logging.info(lang_conf["INFO_EXECUTING"].format(titel=chosen_proposal.get('titel', '[N/A]')))

                # Prepare arguments for the story_generator.py script
                # Ensure all arguments are strings for subprocess
                prompt_arg = str(chosen_proposal.get('prompt', ''))
                setting_arg = str(chosen_proposal.get('setting', ''))
                title_arg = str(chosen_proposal.get('titel', ''))
                wordcount_arg = str(chosen_proposal.get('wortanzahl', DEFAULT_TARGET_WORDS_MIN)) # Use 'wortanzahl'
                # Map language code ('de'/'en') to the full name expected by story_generator
                language_arg_story = lang_conf['CMD_LANG_MAP'].get(args.language_code, args.language_code)

                # Construct the command as a list of arguments for subprocess.run
                command = [
                    lang_conf['CMD_PYTHON_EXECUTABLE'], # e.g., '/usr/bin/python3'
                    lang_conf['CMD_GENERATOR_SCRIPT'], # e.g., 'story_generator.py'
                    "--prompt", prompt_arg,
                    "--setting", setting_arg,
                    "--title", title_arg, # Use --title
                    "--wordcount", wordcount_arg, # Use --wordcount
                    "--language", language_arg_story, # Use --language with full name
                    lang_conf['CMD_SAVE_TEXT_ARG'] # Add --save-text automatically
                    # Optional: Pass --output-dir to the story generator as well
                    # lang_conf['CMD_OUTPUT_DIR_ARG'], args.output_dir,
                ]
                # Pass API key if it was provided to this script
                if args.api_key:
                     command.extend(["--api-key", args.api_key])
                # Pass debug flag if enabled
                if args.debug:
                     command.append("--debug")


                logging.info(lang_conf["INFO_RUNNING_GENERATION"])
                # Log the command being executed (for debugging, be careful with sensitive args like API keys if logging publicly)
                # Creating a display string requires careful handling of quotes, especially for prompt/setting
                # For debugging, it's often safer to print the list `command` directly.
                logging.debug(f"Executing command list: {command}")

                try:
                    # Execute the story generator script as a subprocess
                    process = subprocess.run(
                        command,
                        check=False, # Don't raise exception on non-zero exit code automatically
                        capture_output=True, # Capture stdout/stderr
                        text=True, # Decode output as text
                        encoding='utf-8' # Specify encoding
                    )
                    # Log the output from the subprocess
                    logging.info(f"'{lang_conf['CMD_GENERATOR_SCRIPT']}' STDOUT:\n{process.stdout}")
                    if process.stderr:
                         # Log stderr separately, often used for errors/warnings
                         logging.error(f"'{lang_conf['CMD_GENERATOR_SCRIPT']}' STDERR:\n{process.stderr}")
                    if process.returncode != 0:
                         # Log if the subprocess exited with an error code
                         logging.error(f"'{lang_conf['CMD_GENERATOR_SCRIPT']}' exited with code {process.returncode}")

                except FileNotFoundError:
                    # Handle error if the story generator script itself is not found
                    logging.error(f"Error: The script '{lang_conf['CMD_GENERATOR_SCRIPT']}' was not found.")
                    logging.error("Please ensure it's in the same directory or in the system's PATH.")
                except Exception as sub_e:
                    # Catch any other errors during subprocess execution
                    logging.error(f"Error executing '{lang_conf['CMD_GENERATOR_SCRIPT']}': {sub_e}")


    except ValueError as ve:
        # Catch configuration errors (e.g., invalid language, missing API key during init)
        logging.error(f"Configuration Error: {ve}")
        sys.exit(1)
    except Exception as e:
        # Catch any unexpected errors during the main execution flow
        logging.error(f"An unexpected error occurred: {e}")
        if args.debug:
            # Print the full traceback only if debug mode is enabled
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    # This ensures the script runs the main function when executed directly
    main()
# --- END OF FILE idea_generator.py ---