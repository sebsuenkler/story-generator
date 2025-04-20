# AI Story Generation Suite

This project provides a suite of Python scripts to generate creative story ideas based on genres and then write complete short stories using a Large Language Model (LLM) accessed via the Nebius AI Studio API using [phi-4 by Microsoft](https://huggingface.co/microsoft/phi-4).

## Overview
The suite consists of three main components:

1.  **`idea_generator.py`**: Generates a list of story proposals (title, prompt, setting, genre, target word count) based on user-specified genres.
2.  **`story_generator.py`**: Takes a detailed prompt, setting, title, and word count to generate a full story, optionally splitting it into chapters for longer narratives and focusing on writing quality.
3.  **`app.py`**: Orchestrates the process by first running the idea generator, allowing the user to select a proposal, and then invoking the story generator to write the chosen story.

## Live-Demo

[https://suenkler-ai.de/story-generator](https://suenkler-ai.de/story-generator)

## Features

*   **Idea Generation**: Create multiple unique story ideas based on selected genres (e.g., Sci-Fi, Fantasy, Mystery).
*   **Customizable Ideas**: Specify the number of ideas and target language.
*   **Interactive Selection**: Choose a generated idea to develop further, or select one randomly.
*   **Detailed Story Writing**: Generate full stories from a chosen idea, including prompt, setting, and title.
*   **Chapter Mode**: Automatically splits longer stories into chapters for better structure and handling LLM context limits.
*   **Quality-Focused Prompts**: Uses detailed prompts for the story generator to encourage higher-quality output (style, flow, avoiding repetition, "show don't tell").
*   **Multi-Language Support**: Generate ideas and stories in German (`de`) and English (`en`). UI elements adapt accordingly.
*   **File Saving**:
    *   Generated ideas can be automatically saved as individual `.txt` files.
    *   Generated stories (including plot outlines for chapter mode) are saved as `.txt` files.
*   **Standalone Use**: Both `idea_generator.py` and `story_generator.py` can be run independently.

## Prerequisites

*   **Python**: Version 3.7 or higher recommended.
*   **Nebius AI Studio Account**: You need an account with Nebius AI Studio ([https://nebius.ai/](https://nebius.ai/)) to obtain an API key.
*   **API Key**: A valid Nebius API key is required to interact with the LLM.

## Installation

1.  **Clone or Download:** Get the script files (`app.py`, `idea_generator.py`, `story_generator.py`) and place them in the same directory.
    ```bash
    # If using git
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```
2.  **Install Dependencies:** Install the required Python packages using pip. Create a file named `requirements.txt` in the project directory with the following content:
    # requirements.txt
    ```txt
    openai>=1.0.0,<2.0.0
    python-dotenv>=1.0.0
    ```

    Then run:
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

The scripts require your Nebius API key. You can provide it in one of two ways:

1.  **`.env` File (Recommended):** Create a file named `.env` in the same directory as the scripts and add your API key like this:
    ```dotenv
    # .env
    NEBIUS_API_KEY="YOUR_NEBIUS_API_KEY_HERE"
    ```
    The scripts will automatically load the key from this file. Remember to add `.env` to your `.gitignore` file if using version control.

2.  **Command-Line Argument:** You can pass the API key directly using the `--api-key` argument when running any of the scripts.
    ```bash
    python app.py --api-key "YOUR_NEBIUS_API_KEY_HERE" ...
    ```

**Note:** Using the command-line argument will override the `.env` file if both are present.

## Usage

You can either use the main application for the full workflow or run the individual generator scripts.

### 1. Running the Main Application (`app.py`)

This is the recommended way to use the suite for the combined idea-to-story workflow.

```bash
python app.py [OPTIONS]
```
#### Key Options:
```bash
--lang CODE, --language CODE: Language code for idea generation and story writing (de or en). Default: de.
--genres GENRE [GENRE ...]: Specify genres for idea generation (e.g., --genres "Science Fiction" Fantasy). Defaults to a predefined list if omitted.
--count NUM: Number of story ideas to generate. Default: 3.
--api-key KEY: Your Nebius API key (overrides .env).
--output-dir PATH: Directory to save the final generated story .txt file. Default: ./generated_stories.
--proposals-folder PATH: Folder where idea .txt files are stored (used by --auto-save-ideas). Default: ./proposals.
--auto-save-ideas: Automatically save all generated ideas as .txt files in the proposals folder.
--debug: Enable detailed debug logging.
```
#### Workflow:
The script generates story ideas using `idea_generator.py`.
Generated ideas are displayed.
If `--auto-save-ideas` is used, the ideas are saved to the `--proposals-folder`.
You are prompted to select an idea number, choose randomly (r), or quit (q).
If an idea is selected, the script invokes `story_generator.py's` logic directly.
The final story is generated and saved to the `--output-dir`.

#### Example:

**Generate 5 English fantasy/cyberpunk ideas, auto-save them, then select one to write and save the story to my_output:** 

```bash
python app.py --lang en --genres Fantasy Cyberpunk --count 5 --auto-save-ideas --output-dir ./my_output
```

### 2. Running the Idea Generator Standalone (idea_generator.py)
Use this if you only want to generate story ideas and save their details.

```bash
python idea_generator.py [OPTIONS]
```
#### Key Options:
```bash
--lang CODE: Language code (de or en). Default: de.
--genres GENRE [GENRE ...]: Genres for idea generation.
--count NUM: Number of ideas. Default: 3.
--api-key KEY: Nebius API key.
--model NAME: Specify LLM model name (overrides default).
--output-dir PATH: Directory to save individual idea .txt files (if not using --auto-save). Default: . (current directory).
--proposals-folder PATH: Folder used by --auto-save. Default: ./proposals.
--auto-save: Automatically save all ideas to the proposals folder.
--select: After generating, prompt interactively to choose an idea.
--execute: If an idea is selected (interactively or randomly), attempt to run story_generator.py as a subprocess with the chosen idea's details.
--debug: Enable debug logging.
```

#### Examples:
1. Generate 4 German mystery ideas and save them automatically
```bash
python idea_generator.py --lang de --genres Mystery --count 4 --auto-save
```

2. Generate 2 English Sci-Fi ideas, then select one and execute story_generator.py
```bash
python idea_generator.py --lang en --genres "Science Fiction" --count 2 --select --execute
```

### 3. Running the Story Generator Standalone (story_generator.py)
Use this if you already have a specific prompt, setting, and title and want to generate the story directly.
```bash
python story_generator.py [OPTIONS]
```
#### Key Options:
```bash
--prompt TEXT: Required. The main plot idea or premise.
--setting TEXT: Required. Description of the story's setting (place, time, atmosphere).
--title TEXT: Required. The title for the story.
--wordcount NUM: Target word count for the story. Default: 5000.
--language NAME: Required. Full language name (Deutsch or Englisch).
--additional TEXT: Optional additional instructions for the LLM.
--api-key KEY: Nebius API key.
--model NAME: Specify LLM model name.
--save-text: Save the generated story as a .txt file.
--output-dir PATH: Directory to save the story and outline files. Default: . (current directory).
--no-chapter-mode: Disable automatic chapter splitting for long stories.
--debug: Enable debug logging.
```

#### Example:
**Generate a ~2000 word English story and save it** 
```bash
python story_generator.py \
    --title "The Last Signal" \
    --prompt "A lone astronaut on Mars receives a cryptic signal from an unknown source, just as their life support begins to fail." \
    --setting "Desolate Martian landscape, near future, claustrophobic hab module, growing sense of dread." \
    --wordcount 2000 \
    --language Englisch \
    --save-text \
    --output-dir ./direct_stories
```
