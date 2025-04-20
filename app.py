# --- START OF FILE main_app.py ---

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import logging
import random
from typing import List, Dict, Optional

# --- Import classes from the other scripts ---
# Ensure the other scripts are in the same directory or Python's path
try:
    from idea_generator import PromptSettingGenerator, DEFAULT_PROPOSALS_FOLDER, STORY_GENERATOR_SCRIPT_NAME
    from story_generator import ShortStoryGenerator, MODELL_NAME as STORY_MODELL_NAME # Use specific model name if needed
except ImportError as e:
    print(f"Error: Could not import necessary classes. Make sure 'idea_generator.py' and 'story_generator.py' are in the same directory or accessible via PYTHONPATH.")
    print(f"Details: {e}")
    sys.exit(1)

# Logging configuration for the main app
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MainApp")

# --- Language Mapping ---
# Map language codes used in idea_generator to full names used in story_generator
LANG_CODE_TO_FULL = {
    "de": "Deutsch",
    "en": "Englisch"
}

def display_proposals(proposals: List[Dict], lang_conf: Dict) -> None:
    """Displays the generated proposals to the user."""
    print("\n" + "=" * 30 + " Generated Story Ideas " + "=" * 30)
    if not proposals:
        print("No proposals were generated.")
        return

    for i, proposal in enumerate(proposals, 1):
        title = proposal.get("titel", "[No Title]")
        genre = proposal.get("genre", "[Unknown]")
        wordcount = proposal.get("wortanzahl", "[N/A]")
        prompt_text = proposal.get("prompt", "")
        setting_text = proposal.get("setting", "")

        # Use labels from the loaded language config
        print(lang_conf["INFO_PROPOSAL_SUMMARY"].format(idx=i, titel=title, genre=genre, wortanzahl=wordcount))
        prompt_summary = prompt_text[:150] + "..." if len(prompt_text) > 150 else prompt_text
        setting_summary = setting_text[:100] + "..." if len(setting_text) > 100 else setting_text
        print(f"  {lang_conf['LABEL_PROMPT']}: {prompt_summary}")
        print(f"  {lang_conf['LABEL_SETTING']}: {setting_summary}")
        print("-" * 25)
    print("=" * (60 + len(" Generated Story Ideas ")))


def select_proposal(proposals: List[Dict], lang_conf: Dict) -> Optional[Dict]:
    """Allows the user to interactively select a proposal or choose randomly."""
    if not proposals:
        return None

    max_idx = len(proposals)
    prompt_text = lang_conf["PROMPT_CHOOSE_EXECUTE"].format(max_idx=max_idx) + \
                  " ('r' for random, 'q' to quit): " # Add random option

    while True:
        try:
            choice = input(prompt_text).strip().lower()
            if choice == 'q':
                logger.info("User quit selection.")
                return None
            elif choice == 'r':
                selected_index = random.randint(0, max_idx - 1)
                logger.info(f"Randomly selected proposal: {selected_index + 1}")
                return proposals[selected_index]
            else:
                selected_index = int(choice) - 1
                if 0 <= selected_index < max_idx:
                    logger.info(f"User selected proposal: {selected_index + 1}")
                    return proposals[selected_index]
                else:
                    print(lang_conf["PROMPT_INVALID_RANGE"].format(max_idx=max_idx))
        except ValueError:
            print(lang_conf["PROMPT_INVALID_INPUT"])
        except KeyboardInterrupt:
            logger.info("\nSelection interrupted by user.")
            return None


def run_story_generation(proposal: Dict, language_code: str, api_key: Optional[str], output_dir: str, debug: bool) -> None:
    """
    Initializes and runs the ShortStoryGenerator with the selected proposal.
    """
    logger.info(f"Preparing to generate story for: '{proposal.get('titel', '[N/A]')}'")

    # Map the language code ('de'/'en') to the full name expected by story_generator
    story_language_full = LANG_CODE_TO_FULL.get(language_code)
    if not story_language_full:
        logger.error(f"Internal Error: Cannot map language code '{language_code}' to a full name.")
        return # Cannot proceed without a valid language name

    try:
        # Initialize the story generator
        # Pass API key and potentially model name if different defaults are desired
        story_gen = ShortStoryGenerator(api_key=api_key, model=STORY_MODELL_NAME)
        lang_conf_story = story_gen._get_lang_config(story_language_full) # Get config for logging

        # Extract details from the proposal (use .get for safety)
        title = str(proposal.get('titel', 'Untitled Story'))
        prompt = str(proposal.get('prompt', ''))
        setting = str(proposal.get('setting', ''))
        # Use 'wortanzahl' key from the proposal
        wordcount = int(proposal.get('wortanzahl', ShortStoryGenerator.DEFAULT_TARGET_WORDS))

        # --- Call the story generator method ---
        logger.info(lang_conf_story["INFO_GENERATING_STORY"].format(
            titel=title, wortanzahl=wordcount, sprache=story_language_full
        ))

        # Note: We are calling the generate_story method directly, not using subprocess
        generated_story = story_gen.generate_story(
            prompt=prompt,
            setting=setting,
            title=title,
            word_count=wordcount,
            language=story_language_full,
            additional_instructions=None, # Add option for this later if needed
            chapter_mode=True, # Default to chapter mode for potentially long stories
            max_words_per_chapter=ShortStoryGenerator.MAX_WORDS_PER_CHAPTER
            # Pass debug status indirectly via logging level setup if needed,
            # or add a debug param to generate_story if it influences internal logic beyond logging.
        )

        if generated_story:
            actual_word_count = len(generated_story.split())
            logger.info(lang_conf_story["INFO_FINAL_WORD_COUNT"].format(wortanzahl=actual_word_count))

            # Save the story
            try:
                saved_story_path = story_gen.save_as_text_file(
                    content=generated_story,
                    title=title,
                    language=story_language_full,
                    output_path=output_dir # Save in the specified output directory
                )
                logger.info(f"Story successfully generated and saved to: {saved_story_path}")
                # Optionally print the story content as well
                # print("\n--- Generated Story ---")
                # print(generated_story)
                # print("--- End Story ---")

            except Exception as save_err:
                logger.error(f"Failed to save the generated story: {save_err}")
                # Print story to console as fallback if saving failed
                print("\n--- Generated Story (Save Failed) ---")
                print(generated_story)
                print("--- End Story ---")
        else:
            logger.error(lang_conf_story["ERROR_GENERATION_FAILED"])

    except ValueError as ve:
        # Catch config errors from story generator init (e.g., language)
        logger.error(f"Story Generator Configuration Error: {ve}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during story generation: {e}")
        if debug:
            import traceback
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Main application to generate story ideas and then write the selected story.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # --- Arguments for Idea Generation ---
    parser.add_argument("--genres", type=str, nargs="*",
                        help="Genres/themes for idea generation (multiple possible)")
    parser.add_argument("--language", "--lang", type=str, default="en", choices=["de", "en"],
                        dest="language_code",
                        help="Language code for generating ideas AND the final story (de/en)")
    parser.add_argument("--count", type=int, default=3, dest="proposal_count",
                        help="Number of story ideas to generate")
    parser.add_argument("--proposals-folder", type=str, default=DEFAULT_PROPOSALS_FOLDER,
                        help="Folder to auto-save generated idea files")
    parser.add_argument("--auto-save-ideas", action="store_true",
                        help="Automatically save all generated idea files to the proposals folder")

    # --- Arguments for Story Generation (and shared) ---
    parser.add_argument("--api-key", type=str, default=os.environ.get("NEBIUS_API_KEY"), # Read from env by default
                        help="Nebius API key (reads NEBIUS_API_KEY env var if not set)")
    parser.add_argument("--output-dir", type=str, default="generated_stories",
                        help="Directory for saving the final generated story text file")
    parser.add_argument("--debug", action="store_true", help="Enable detailed debug logging for all steps")

    args = parser.parse_args()

    # --- Configure Logging Level ---
    log_level = logging.DEBUG if args.debug else logging.INFO
    # Apply level to the main app logger and potentially imported module loggers if needed
    logger.setLevel(log_level)
    logging.getLogger("IdeaGenerator").setLevel(log_level) # Set level for idea_generator logs
    logging.getLogger("ShortStoryGenerator").setLevel(log_level) # Set level for story_generator logs
    # Reconfigure root logger if necessary (can affect library logs)
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', force=True)
    logger.debug("Debug mode enabled for MainApp.")

    # --- Validate API Key ---
    if not args.api_key:
        logger.error("API Key is required. Please provide --api-key or set the NEBIUS_API_KEY environment variable.")
        sys.exit(1)

    # --- Step 1: Generate Story Ideas ---
    logger.info("--- Step 1: Generating Story Ideas ---")
    idea_gen = None
    selected_proposal = None
    try:
        idea_gen = PromptSettingGenerator(
            api_key=args.api_key,
            proposals_folder=args.proposals_folder
            # Add model override if needed: model=args.idea_model
        )
        # Get language config for UI messages during idea generation/selection
        lang_conf_idea = idea_gen._get_lang_config(args.language_code)

        # Prepare genres (using logic similar to idea_generator.py main)
        default_genres = ["Sherlock Holmes", "Science Fiction", "Fantasy", "Horror", "Mystery", "Dark Fantasy", "Cyberpunk"]
        if not args.genres:
            args.genres = default_genres
        selected_genres = args.genres
        if len(args.genres) > 3:
            k = random.randint(1, 3)
            selected_genres = random.sample(args.genres, k=k)
        logger.info(f"Using genres for idea generation: {', '.join(selected_genres)}")

        # Generate proposals
        proposals = idea_gen.generate_proposals(
            genres=selected_genres,
            count=args.proposal_count,
            language_code=args.language_code
        )

        if not proposals:
            logger.error("Failed to generate any story ideas. Exiting.")
            sys.exit(1)

        # Optionally auto-save ideas
        if args.auto_save_ideas:
            logger.info(f"Auto-saving generated ideas to: {args.proposals_folder}")
            required_keys_for_saving = ["titel", "prompt", "setting", "genre", "wortanzahl"]
            for i, prop in enumerate(proposals, 1):
                missing_keys = [k for k in required_keys_for_saving if k not in prop or not prop[k]]
                if missing_keys:
                    logger.warning(lang_conf_idea["WARN_INCOMPLETE_PROPOSAL"].format(idx=i, missing=', '.join(missing_keys)))
                    continue
                try:
                    idea_gen.save_proposal_as_txt(prop, args.language_code, use_proposals_folder=True)
                except Exception as e:
                    logger.error(lang_conf_idea["ERROR_SAVING_PROPOSAL"].format(idx=i, error=e))


        # --- Step 2: Display and Select Idea ---
        logger.info("--- Step 2: Select a Story Idea ---")
        display_proposals(proposals, lang_conf_idea)
        selected_proposal = select_proposal(proposals, lang_conf_idea)

    except ValueError as ve:
        logger.error(f"Configuration Error during idea generation: {ve}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred during idea generation/selection: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


    # --- Step 3: Generate the Story ---
    if selected_proposal:
        logger.info("--- Step 3: Generating Selected Story ---")
        # Ensure the output directory for the final story exists
        try:
            os.makedirs(args.output_dir, exist_ok=True)
            logger.info(f"Ensured output directory for story exists: {args.output_dir}")
        except OSError as e:
            logger.error(f"Could not create output directory '{args.output_dir}': {e}. Exiting.")
            sys.exit(1)

        run_story_generation(
            proposal=selected_proposal,
            language_code=args.language_code, # Use the same language code
            api_key=args.api_key,
            output_dir=args.output_dir,
            debug=args.debug
        )
    else:
        logger.info("No proposal selected or user quit. Story generation skipped.")

    logger.info("Main application finished.")


if __name__ == "__main__":
    main()
# --- END OF FILE main_app.py ---