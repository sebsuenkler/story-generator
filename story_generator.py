# --- START OF FILE story_generator.py ---

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import json
import datetime
import time
import math
import re
import logging
import tempfile
import random # for jitter in retry
from typing import List, Tuple, Optional, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# === Configuration and Constants ===
MODELL_NAME = "microsoft/phi-4" # Or another model that follows instructions well
API_BASE_URL = "https://api.studio.nebius.com/v1/"

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ShortStoryGenerator: # Renamed class for clarity
    """
    Generates short stories with enhanced quality instructions in the prompt
    to minimize the need for subsequent editing.
    """
    # --- Constants for the class ---
    # API & Retry
    DEFAULT_RETRY_DELAY_S: int = 10
    MAX_RETRIES: int = 3
    RETRY_BACKOFF_FACTOR: float = 1.5
    # Word count & Token
    MIN_STORY_WORDS: int = 500
    MAX_STORY_WORDS_NO_CHAPTERS: int = 25000
    DEFAULT_TARGET_WORDS: int = 5000
    TOKEN_WORD_RATIO: float = 1.6 # Set slightly higher for potentially more detailed text
    MAX_TOKENS_PER_CALL: int = 15000 # Adjusted based on typical model limits
    MAX_WORDS_PER_CHAPTER: int = 3000
    # Chapter & Structure
    MIN_CHAPTERS_LONG_STORY: int = 3
    TARGET_WORDS_PER_CHAPTER_DIVISOR: int = 2500
    WORD_COUNT_BUFFER_FACTOR: float = 2 # Reduced buffer as prompts are more directive
    # Text cleanup
    MIN_WORDS_FOR_VALID_ENDING: int = 4
    MIN_CHARS_FOR_RESCUE: int = 100
    # --- ADJUSTED Language configuration with QUALITY INSTRUCTIONS ---
    SUPPORTED_LANGUAGES: List[str] = ["Deutsch", "Englisch"] # Keep full names for clarity
    LANGUAGE_CONFIG: Dict[str, Dict[str, Any]] = {
        "Deutsch": {
            # --- Prompt for single story (revised) ---
            "PROMPT_STORY_GEN": """Du bist ein talentierter und erfahrener Autor von Kurzgeschichten in **deutscher Sprache**. Deine Aufgabe ist es, eine fesselnde, gut geschriebene Geschichte basierend auf den folgenden Vorgaben zu erstellen.

Ziel-Wortanzahl: ca. {wortanzahl} Wörter.

Titel: {titel}
Prompt (Grundidee): {prompt}
Setting (Ort, Zeit, Atmosphäre): {setting}

**Qualitätsrichtlinien für das Schreiben:**
-   **Stil & Fluss:** Schreibe in klarem, prägnantem und ansprechendem Deutsch. Variiere Satzlänge und -struktur, um Monotonie zu vermeiden. Sorge für einen natürlichen Lesefluss.
-   **Wiederholungen VERMEIDEN:** Achte SEHR GENAU darauf, Wort- und Phrasenwiederholungen zu minimieren. Nutze Synonyme oder formuliere Sätze kreativ um, ohne den Sinn zu verändern.
-   **Starke Sprache:** Verwende präzise Verben und Adjektive. Bevorzuge aktive Formulierungen gegenüber passiven.
-   **"Show, Don't Tell":** Zeige Emotionen, Charaktereigenschaften und Atmosphäre durch Handlungen, Dialoge und sensorische Details, anstatt sie nur zu behaupten.
-   **Dialog:** Schreibe glaubwürdige, lebendige Dialoge, die zur jeweiligen Figur passen und die Handlung vorantreiben oder Charaktertiefe verleihen.
-   **Setting & Atmosphäre:** Beschreibe den Ort, die Zeit und die Stimmung lebendig und immersiv. Nutze Details, die die Sinne ansprechen.
-   **Pacing & Spannung:** Baue die Handlung logisch auf, erzeuge Spannung und variiere das Tempo angemessen.
-   **Kohärenz & Konsistenz:** Erzähle eine in sich schlüssige Geschichte mit klarem Anfang, Mittelteil und Ende. Achte auf Konsistenz bei Zeitformen, Perspektive und Charakterdetails.
-   **Abgeschlossenes Ende:** Die Geschichte MUSS ein klares, befriedigendes und abgeschlossenes Ende haben. Sie darf NICHT abrupt oder mitten im Satz/Absatz enden.
-   **Formatierung:** Beginne NUR mit dem Titel im Format '# {titel}'. Füge KEINEN weiteren Titel hinzu. Strukturiere den Text sinnvoll mit Absätzen.

{zusatz_anweisungen}""",
            # --- Prompt for outline (slightly adjusted) ---
            "PROMPT_OUTLINE_GEN": """Du bist ein erfahrener Autor und Story-Entwickler. Erstelle eine detaillierte Plot-Outline für eine Geschichte mit {kapitel_anzahl} Kapiteln basierend auf:
Titel: {titel}
Prompt: {prompt}
Setting: {setting}
Gesamtwortzahl: ca. {wortanzahl} Wörter ({sprache})

Erstelle eine Outline mit:
1.  Hauptcharaktere: Details (Name, Alter, Aussehen, Persönlichkeit, Motivation).
2.  Wichtige Nebencharaktere: Namen, Beziehung.
3.  Haupthandlung: Detaillierte Zusammenfassung des Verlaufs.
4.  Weltregeln (falls relevant).
5.  Kapitelstruktur ({kapitel_anzahl} Kapitel): Für jedes Kapitel: Titel, Zeit/Ort, Charaktere, Handlungspunkte (5-7), Spannungsbogen/Wendungen, Cliffhanger (optional).

**Wichtig:** Die Outline soll eine Grundlage für eine **qualitativ hochwertige Geschichte** bilden. Achte auf Konsistenz, Kausalität, Charakterentwicklung und Potenzial für spannendes Erzählen gemäß hoher Schreibstandards (abwechslungsreiche Sprache, Vermeidung von Wiederholungen im Konzept etc.).
{zusatz_anweisungen}""",
             # --- Prompt for chapter (heavily revised) ---
            "PROMPT_CHAPTER_GEN": """Du bist ein talentierter und erfahrener Autor von Kurzgeschichten in **deutscher Sprache**. Deine Aufgabe ist es, **Kapitel {kapitel_nummer} von {kapitel_anzahl}** einer Geschichte herausragend zu schreiben, basierend auf Titel, Prompt, Setting und der Plot-Outline.

Gesamttitel: {titel}
Prompt (Gesamtidee): {prompt}
Setting (Gesamt): {setting}
Plot-Outline (Gesamt): {plot_outline}
{zusammenfassung_vorher}

**Anweisungen für DIESES Kapitel ({kapitel_nummer}):**
-   **Inhalt:** Halte dich eng an die Handlungselemente, Charakterentwicklung und den Zeitrahmen, die in der Plot-Outline für **dieses spezifische Kapitel** vorgesehen sind.
-   **Umfang:** Dieses Kapitel sollte ca. {kapitel_wortanzahl} Wörter haben (mind. {min_kapitel_worte}, max. {max_kapitel_worte}).
-   **Sprache & Stil:** Schreibe in klarem, prägnantem und ansprechendem Deutsch. Variiere Satzlänge und -struktur. Sorge für einen exzellenten Lesefluss.
-   **Wiederholungen STRENG VERMEIDEN:** Achte extrem darauf, Wort- und Phrasenwiederholungen innerhalb des Kapitels und im Vergleich zum *direkt vorherigen* Kapitel (falls Kontext gegeben) zu vermeiden. Nutze Synonyme und kreative Umformulierungen.
-   **Starke Sprache:** Nutze präzise, aktive Verben und lebendige Adjektive.
-   **"Show, Don't Tell":** Zeige Emotionen, Atmosphäre etc. durch konkrete Beschreibung und Handlung.
-   **Dialog:** Schreibe glaubwürdige, figurengerechte Dialoge.
-   **Atmosphäre:** Baue die im Setting beschriebene Stimmung auf.
-   **Pacing:** Gestalte den Spannungsbogen dieses Kapitels passend zur Outline.
-   **Konsistenz:** Stelle absolute Kontinuität zur Plot-Outline und den vorherigen Kapiteln her (Charaktere, Ereignisse, Zeit).
-   **Übergänge:** Sorge für einen **fließenden Übergang** vom Ende des vorherigen Kapitels (siehe Kontext unten, falls vorhanden) zum Anfang dieses Kapitels. Stelle sicher, dass das Ende dieses Kapitels logisch abschließt oder sinnvoll zum nächsten Kapitel überleitet (siehe Kontext unten, falls vorhanden).
-   **Formatierung:** Beginne mit der Kapitelüberschrift: '# Kapitel {kapitel_nummer}: [Passender Kapiteltitel aus Outline oder neu generiert]' (AUSSER Kapitel 1: Beginne mit '# {titel}\\n\\n# Kapitel 1: [Kapiteltitel]'). Füge KEINEN Titel am Ende hinzu.
-   **Abschluss:** Das Kapitel MUSS mit einem vollständigen, sinnvollen Satz und Absatz enden. Kein Abbruch mitten im Satz/Wort/Gedanken.

{zusatz_anweisungen}""",
            # --- Prompt for epilogue (revised) ---
            "PROMPT_EPILOG_GEN": """Du bist ein talentierter Autor. Schreibe einen **hochwertigen, gut geschriebenen** und befriedigenden Abschluss (Epilog) für eine Geschichte.
Titel: {titel}
Plot-Outline: {plot_outline}
Zusammenfassung bis zum letzten Kapitel: {zusammenfassung_vorher}
Möglicherweise unvollständiges Ende des letzten Kapitels:
...{letztes_kapitel_ende}

Schreibe einen Abschluss (ca. 500-1000 Wörter) in **Deutsch**, der:
1.  Offene Handlungsstränge elegant auflöst.
2.  Die emotionale Reise der Charaktere zu einem sinnvollen Ende führt.
3.  Die Hauptthemen aufgreift.
4.  Stilistisch zum Rest der Geschichte passt (abwechslungsreiche Sprache, keine Wiederholungen).
5.  Ein klares, befriedigendes Ende hat.

Beginne mit '# Epilog'. Vermeide Wiederholungen aus dem letzten Kapitel.
{zusatz_anweisungen}""",
            # --- Other texts ---
            "USER_PROMPT_STORY": "Bitte schreibe eine hochwertige Kurzgeschichte basierend auf Titel '{titel}', Prompt und Setting. Ca. {wortanzahl} Wörter.",
            "USER_PROMPT_OUTLINE": "Bitte erstelle eine detaillierte Plot-Outline für eine {kapitel_anzahl}-teilige, qualitativ hochwertige Geschichte mit Titel '{titel}'.",
            "USER_PROMPT_CHAPTER": "Bitte schreibe Kapitel {kapitel_nummer} der Geschichte '{titel}' herausragend und gemäß allen Qualitätsrichtlinien.",
            "USER_PROMPT_EPILOG": "Bitte schreibe einen hochwertigen Epilog für die Geschichte '{titel}'.",
            "DEFAULT_CHAPTER_TITLE": "Kapitel {kapitel_nummer}: Fortsetzung",
            "FIRST_CHAPTER_TITLE": "Kapitel 1: Einleitung",
            "EPILOG_TITLE": "Epilog",
            "ERROR_CHAPTER_TITLE": "Kapitel {kapitel_nummer}: [Fehler bei der Generierung]",
            "ERROR_CHAPTER_CONTENT": "Es ist ein Fehler bei der Generierung dieses Kapitels aufgetreten.",
            "RESCUED_CHAPTER_NOTICE": "[Das Kapitel wurde aufgrund technischer Probleme gekürzt. Die Geschichte wird im nächsten Kapitel fortgesetzt.]",
            "RESCUED_EPILOG_NOTICE": "[Die Geschichte wurde nicht vollständig generiert. Bitte wenden Sie sich an den Autor für den Abschluss.]",
            "SUMMARY_FIRST_CHAPTER": "Dies ist das erste Kapitel.",
            "SUMMARY_PREVIOUS_CHAPTERS_INTRO": "Kontext - Zusammenfassung der bisherigen Handlung:",
            "SUMMARY_CHAPTER_PREFIX": "Kapitel {num}:",
            "SUMMARY_LAST_CHAPTER_PREFIX": "Zuletzt in Kapitel {num} (Kontext):",
            "WARN_LOW_WORDCOUNT": "Warnung: Wortanzahl {wortanzahl} sehr niedrig. Setze auf Minimum {min_val}.",
            "WARN_HIGH_WORDCOUNT": "Warnung: Wortanzahl {wortanzahl} zu hoch ohne Kapitel-Modus. Setze auf Maximum {max_val}.",
            "INFO_CHAPTER_MODE_ACTIVATED": "Aktiviere Kapitel-Modus für Geschichte mit {wortanzahl} Wörtern.",
            "INFO_GENERATING_STORY": "Generiere Geschichte '{titel}' mit ca. {wortanzahl} Wörtern in {sprache} (Qualitätsfokus)...",
            "INFO_SENDING_API_REQUEST": "Sende Anfrage an API (max_tokens={max_tokens}, temp={temp:.1f})...",
            "INFO_WAITING_API_RESPONSE": "Warte auf API-Antwort (dies kann einige Minuten dauern)...",
            "INFO_GENERATION_COMPLETE": "Generierung abgeschlossen nach {dauer:.1f} Sekunden.",
            "INFO_SAVED_TEMP_FILE": "Daten in temporäre Datei gespeichert: {dateiname}",
            "WARN_CANNOT_WRITE_TEMP": "Warnung: Konnte temporäre Datei nicht schreiben: {error}",
            "WARN_CANNOT_DELETE_TEMP": "Warnung: Konnte temporäre Datei nicht löschen: {error}",
            "INFO_REMOVED_INCOMPLETE_SENTENCE": "Unvollständiger Satz am Ende entfernt.",
            "INFO_CORRECTING_ENDING": "Ende wird korrigiert...",
            "INFO_REMOVED_INCOMPLETE_DIALOG": "Unvollständigen Dialog am Ende entfernt.",
            "INFO_ADDED_MISSING_QUOTE": "Fehlendes Anführungszeichen ergänzt.",
            "INFO_REMOVED_INCOMPLETE_PARAGRAPH": "Unvollständigen letzten Absatz entfernt.",
            "INFO_GENERATING_CHAPTERS": "Generiere Geschichte in {kapitel_anzahl} Kapiteln (Qualitätsfokus)...",
            "INFO_GENERATING_OUTLINE": "Generiere Plot-Outline für die gesamte Geschichte...",
            "INFO_OUTLINE_CREATED": "Plot-Outline erstellt.",
            "INFO_SAVED_OUTLINE": "Plot-Outline gespeichert als: {dateiname}",
            "WARN_CANNOT_SAVE_OUTLINE": "Warnung: Konnte Plot-Outline nicht speichern: {error}",
            "ERROR_SAVING_OUTLINE": "Fehler beim Speichern der Plot-Outline: {error}. Generierung wird fortgesetzt.",
            "INFO_GENERATING_CHAPTER_NUM": "Generiere Kapitel {kapitel_nummer} von {kapitel_anzahl}...",
            "INFO_CHAPTER_COMPLETE": "Kapitel {kapitel_nummer} abgeschlossen nach {dauer:.1f} Sekunden.",
            "ERROR_API_REQUEST_CHAPTER": "Fehler bei der API-Anfrage für Kapitel {kapitel_nummer}: {error}",
            "INFO_RESCUED_PARTIAL_CONTENT": "Teilweise generierter Inhalt ({chars} Zeichen) gerettet.",
            "ERROR_READING_TEMP_FILE": "Fehler beim Lesen der temporären Datei: {error}",
            "INFO_LAST_CHAPTER_INCOMPLETE": "Das letzte Kapitel scheint unvollständig zu sein. Generiere einen Epilog...",
            "INFO_GENERATING_EPILOG": "Generiere Epilog:",
            "INFO_EPILOG_GENERATED": "Epilog generiert.",
            "ERROR_GENERATING_EPILOG": "Fehler bei der Generierung des Epilogs: {error}",
            "INFO_FINAL_WORD_COUNT": "Geschichte mit {wortanzahl} Wörtern generiert.",
            "INFO_SAVED_TEXT_FILE": "Geschichte als Textdatei gespeichert: {dateiname}",
            "ERROR_API_OVERLOAD": "API überlastet/Timeout. Warte {delay:.1f} Sekunden, versuche erneut ({retries}/{max_retries})...",
            "ERROR_API_CALL_FAILED": "Fehler während API-Aufruf: {error}",
            "ERROR_ALL_RETRIES_FAILED": "Alle API-Wiederholungen fehlgeschlagen.",
            "ERROR_UNSUPPORTED_LANGUAGE": "Sprache '{sprache}' nicht unterstützt. Unterstützt: {supported}",
            "ERROR_MISSING_API_KEY": "API-Schlüssel erforderlich. NEBIUS_API_KEY setzen oder übergeben.",
            "ERROR_GENERATION_FAILED": "Es konnte keine Geschichte generiert werden.",
            "COMMON_NOUN_PREFIXES": ["Der", "Die", "Das", "Ein", "Eine"],
            "ACTION_VERBS": ["ging", "kam", "sprach", "sah", "fand", "entdeckte", "öffnete", "schloss", "rannte", "floh", "kämpfte", "starb", "tötete", "küsste", "sagte", "antwortete", "erwiderte", "blickte", "dachte"],
            "EMOTIONAL_WORDS": ["angst", "furcht", "freude", "glück", "trauer", "wut", "zorn", "liebe", "hass", "entsetzen", "überraschung", "schock", "verzweiflung", "erleichterung"],
            "CONJUNCTIONS_AT_END": ['und', 'aber', 'oder', 'denn', 'weil', 'dass', 'ob']
        },
        "Englisch": {
             # --- Prompt for single story (revised) ---
            "PROMPT_STORY_GEN": """You are a talented and experienced author of short stories in **English**. Your task is to create a compelling, well-written story based on the following specifications.

Target word count: approx. {wortanzahl} words.

Title: {titel}
Prompt (Basic Idea): {prompt}
Setting (Location, Time, Atmosphere): {setting}

**Quality Guidelines for Writing:**
-   **Style & Flow:** Write in clear, concise, and engaging English. Vary sentence length and structure to avoid monotony. Ensure a natural reading flow.
-   **AVOID Repetition:** Pay VERY CLOSE attention to minimizing word and phrase repetitions. Use synonyms or creatively rephrase sentences without altering the meaning.
-   **Strong Language:** Use precise verbs and evocative adjectives. Prefer active voice over passive.
-   **"Show, Don't Tell":** Demonstrate emotions, character traits, and atmosphere through actions, dialogue, and sensory details, rather than just stating them.
-   **Dialogue:** Write believable, vivid dialogue that fits the character and either advances the plot or reveals personality.
-   **Setting & Atmosphere:** Describe the location, time, and mood vividly and immersively. Use details that appeal to the senses.
-   **Pacing & Suspense:** Structure the plot logically, build suspense, and vary the pace appropriately.
-   **Coherence & Consistency:** Tell a cohesive story with a clear beginning, middle, and end. Ensure consistency in tense, perspective, and character details.
-   **Complete Ending:** The story MUST have a clear, satisfying, and conclusive ending. It must NOT end abruptly or mid-sentence/paragraph.
-   **Formatting:** Start ONLY with the title in the format '# {titel}'. Do NOT add another title. Structure the text meaningfully with paragraphs.

{zusatz_anweisungen}""",
            # --- Prompt for outline (slightly adjusted) ---
            "PROMPT_OUTLINE_GEN": """You are a seasoned author and story developer. Create a detailed plot outline for a story with {kapitel_anzahl} chapters based on:
Title: {titel}
Prompt: {prompt}
Setting: {setting}
Total word count: approx. {wortanzahl} words ({sprache})

Create an outline including:
1.  Main Characters: Details (name, age, appearance, personality, motivation).
2.  Important Side Characters: Names, relationships.
3.  Main Plot: Detailed summary of the storyline.
4.  World Rules (if applicable).
5.  Chapter Structure ({kapitel_anzahl} chapters): For each chapter: Title, Time/Location, Characters, Plot Points (5-7), Arc/Twists, Cliffhanger (optional).

**Important:** The outline should serve as a foundation for a **high-quality story**. Ensure consistency, causality, character development, and potential for engaging narrative adhering to high writing standards (varied language, avoiding repetition in the concept, etc.).
{zusatz_anweisungen}""",
            # --- Prompt for chapter (heavily revised) ---
            "PROMPT_CHAPTER_GEN": """You are a talented and experienced author of short stories in **English**. Your task is to write **Chapter {kapitel_nummer} of {kapitel_anzahl}** of a story exceptionally well, based on the title, prompt, setting, and plot outline.

Overall Title: {titel}
Prompt (Overall Idea): {prompt}
Setting (Overall): {setting}
Plot Outline (Overall): {plot_outline}
{zusammenfassung_vorher}

**Instructions for THIS Chapter ({kapitel_nummer}):**
-   **Content:** Adhere closely to the plot elements, character development, and timeframe specified in the plot outline for **this specific chapter**.
-   **Length:** This chapter should be approximately {kapitel_wortanzahl} words (min {min_kapitel_worte}, max {max_kapitel_worte}).
-   **Language & Style:** Write in clear, concise, and engaging English. Vary sentence length and structure. Ensure excellent reading flow.
-   **STRICTLY AVOID Repetition:** Be extremely careful to avoid repeating words and phrases within the chapter and compared to the *immediately preceding* chapter (if context is provided). Use synonyms and creative rephrasing.
-   **Strong Language:** Use precise, active verbs and vivid adjectives.
-   **"Show, Don't Tell":** Show emotions, atmosphere, etc., through concrete description and action.
-   **Dialogue:** Write believable dialogue appropriate for each character.
-   **Atmosphere:** Build the mood described in the setting.
-   **Pacing:** Shape the narrative arc of this chapter according to the outline.
-   **Consistency:** Ensure absolute continuity with the plot outline and previous chapters (characters, events, time).
-   **Transitions:** Ensure a **smooth transition** from the end of the previous chapter (see context below, if provided) to the beginning of this one. Ensure the end of this chapter concludes logically or leads effectively into the next chapter (see context below, if provided).
-   **Formatting:** Start with the chapter heading: '# Chapter {kapitel_nummer}: [Appropriate Chapter Title from Outline or newly generated]' (EXCEPT Chapter 1: Start with '# {titel}\\n\\n# Chapter 1: [Chapter Title]'). Do NOT add a title at the end.
-   **Conclusion:** The chapter MUST end with a complete, meaningful sentence and paragraph. No cut-off sentences, words, or thoughts.

{zusatz_anweisungen}""",
            # --- Prompt for epilogue (revised) ---
            "PROMPT_EPILOG_GEN": """You are a talented author. Write a **high-quality, well-written,** and satisfying conclusion (Epilogue) for a story.
Title: {titel}
Plot Outline: {plot_outline}
Summary up to the last chapter: {zusammenfassung_vorher}
Possibly incomplete end of the last chapter:
...{letztes_kapitel_ende}

Write a conclusion (approx. 500-1000 words) in **English** that:
1.  Elegantly resolves open plot threads.
2.  Brings the main characters' emotional journey to a meaningful end.
3.  Revisits main themes.
4.  Matches the style of the rest of the story (varied language, no repetition).
5.  Has a clear, satisfying ending.

Start with '# Epilogue'. Avoid repeating content from the last chapter.
{zusatz_anweisungen}""",
            # --- Other texts ---
            "USER_PROMPT_STORY": "Please write a high-quality short story based on title '{titel}', prompt, and setting. Approx. {wortanzahl} words.",
            "USER_PROMPT_OUTLINE": "Please create a detailed plot outline for a {kapitel_anzahl}-chapter, high-quality story titled '{titel}'.",
            "USER_PROMPT_CHAPTER": "Please write chapter {kapitel_nummer} of the story '{titel}' exceptionally well, following all quality guidelines.",
            "USER_PROMPT_EPILOG": "Please write a high-quality epilogue for the story '{titel}'.",
            "DEFAULT_CHAPTER_TITLE": "Chapter {kapitel_nummer}: Continuation",
            "FIRST_CHAPTER_TITLE": "Chapter 1: Introduction",
            "EPILOG_TITLE": "Epilogue",
            "ERROR_CHAPTER_TITLE": "Chapter {kapitel_nummer}: [Generation Error]",
            "ERROR_CHAPTER_CONTENT": "An error occurred during the generation of this chapter.",
            "RESCUED_CHAPTER_NOTICE": "[This chapter was shortened due to technical issues. The story continues in the next chapter.]",
            "RESCUED_EPILOG_NOTICE": "[The story was not fully generated. Please contact the author for the conclusion.]",
            "SUMMARY_FIRST_CHAPTER": "This is the first chapter.",
            "SUMMARY_PREVIOUS_CHAPTERS_INTRO": "Context - Summary of the story so far:",
            "SUMMARY_CHAPTER_PREFIX": "Chapter {num}:",
            "SUMMARY_LAST_CHAPTER_PREFIX": "Previously in Chapter {num} (Context):",
            "WARN_LOW_WORDCOUNT": "Warning: Word count {wortanzahl} very low. Setting to minimum {min_val}.",
            "WARN_HIGH_WORDCOUNT": "Warning: Word count {wortanzahl} too high without chapter mode. Setting to maximum {max_val}.",
            "INFO_CHAPTER_MODE_ACTIVATED": "Activating chapter mode for story with {wortanzahl} words.",
            "INFO_GENERATING_STORY": "Generating story '{titel}' with approx. {wortanzahl} words in {sprache} (Quality Focus)...",
            "INFO_SENDING_API_REQUEST": "Sending request to API (max_tokens={max_tokens}, temp={temp:.1f})...",
            "INFO_WAITING_API_RESPONSE": "Waiting for API response (this may take several minutes)...",
            "INFO_GENERATION_COMPLETE": "Generation completed in {dauer:.1f} seconds.",
            "INFO_SAVED_TEMP_FILE": "Data saved to temporary file: {dateiname}",
            "WARN_CANNOT_WRITE_TEMP": "Warning: Could not write temporary file: {error}",
            "WARN_CANNOT_DELETE_TEMP": "Warning: Could not delete temporary file: {error}",
            "INFO_REMOVED_INCOMPLETE_SENTENCE": "Removed incomplete sentence at the end.",
            "INFO_CORRECTING_ENDING": "Correcting ending...",
            "INFO_REMOVED_INCOMPLETE_DIALOG": "Removed incomplete dialogue at the end.",
            "INFO_ADDED_MISSING_QUOTE": "Added missing quotation mark.",
            "INFO_REMOVED_INCOMPLETE_PARAGRAPH": "Removed incomplete last paragraph.",
            "INFO_GENERATING_CHAPTERS": "Generating story in {kapitel_anzahl} chapters (Quality Focus)...",
            "INFO_GENERATING_OUTLINE": "Generating plot outline for the entire story...",
            "INFO_OUTLINE_CREATED": "Plot outline created.",
            "INFO_SAVED_OUTLINE": "Plot outline saved as: {dateiname}",
            "WARN_CANNOT_SAVE_OUTLINE": "Warning: Could not save plot outline: {error}",
            "ERROR_SAVING_OUTLINE": "Error saving plot outline: {error}. Continuing generation.",
            "INFO_GENERATING_CHAPTER_NUM": "Generating chapter {kapitel_nummer} of {kapitel_anzahl}...",
            "INFO_CHAPTER_COMPLETE": "Chapter {kapitel_nummer} completed in {dauer:.1f} seconds.",
            "ERROR_API_REQUEST_CHAPTER": "Error in API request for chapter {kapitel_nummer}: {error}",
            "INFO_RESCUED_PARTIAL_CONTENT": "Rescued partially generated content ({chars} characters).",
            "ERROR_READING_TEMP_FILE": "Error reading temporary file: {error}",
            "INFO_LAST_CHAPTER_INCOMPLETE": "The last chapter seems incomplete. Generating an epilogue...",
            "INFO_GENERATING_EPILOG": "Generating epilogue:",
            "INFO_EPILOG_GENERATED": "Epilogue generated.",
            "ERROR_GENERATING_EPILOG": "Error generating epilogue: {error}",
            "INFO_FINAL_WORD_COUNT": "Generated story with {wortanzahl} words.",
            "INFO_SAVED_TEXT_FILE": "Story saved as text file: {dateiname}",
            "ERROR_API_OVERLOAD": "API overloaded/timeout. Wait {delay:.1f} seconds, retry ({retries}/{max_retries})...",
            "ERROR_API_CALL_FAILED": "Error during API call: {error}",
            "ERROR_ALL_RETRIES_FAILED": "All API retries failed.",
            "ERROR_UNSUPPORTED_LANGUAGE": "Language '{sprache}' not supported. Supported: {supported}",
            "ERROR_MISSING_API_KEY": "API key required. Set NEBIUS_API_KEY or pass directly.",
            "ERROR_GENERATION_FAILED": "Could not generate the story.",
            "COMMON_NOUN_PREFIXES": ["The", "A", "An"],
            "ACTION_VERBS": ["went", "came", "spoke", "said", "answered", "replied", "saw", "found", "discovered", "opened", "closed", "ran", "fled", "fought", "died", "killed", "kissed", "looked", "thought", "realized"],
            "EMOTIONAL_WORDS": ["fear", "joy", "happiness", "sadness", "anger", "wrath", "love", "hate", "horror", "surprise", "shock", "despair", "relief"],
            "CONJUNCTIONS_AT_END": ['and', 'but', 'or', 'so', 'yet', 'because', 'if', 'that', 'when', 'while', 'although']
        }
    }

    def __init__(self, api_key: Optional[str] = None, model: str = MODELL_NAME):
        """Initializes the ShortStoryGenerator."""
        # Use German language config for initialization errors as default (can be changed)
        lang_conf_init = self.LANGUAGE_CONFIG["Deutsch"]
        resolved_api_key = api_key or os.environ.get("NEBIUS_API_KEY")
        if not resolved_api_key:
            raise ValueError(lang_conf_init["ERROR_MISSING_API_KEY"])

        self.client = OpenAI(base_url=API_BASE_URL, api_key=resolved_api_key)
        self.model_name = model

    def _get_lang_config(self, language: str) -> Dict[str, Any]:
        """Gets the configuration for the specified language."""
        config = self.LANGUAGE_CONFIG.get(language)
        if not config:
             supported = ", ".join(self.SUPPORTED_LANGUAGES)
             # Use German for the error message itself, or default to English if German is missing
             lang_conf_err = self.LANGUAGE_CONFIG.get("Deutsch", self.LANGUAGE_CONFIG.get("Englisch"))
             raise ValueError(lang_conf_err["ERROR_UNSUPPORTED_LANGUAGE"].format(sprache=language, supported=supported))
        return config

    def _safe_filename(self, title: str) -> str:
        """Creates a safe filename from a title."""
        safe_title = re.sub(r'[^\w\s-]', '', title).strip()
        safe_title = re.sub(r'\s+', '_', safe_title)
        return safe_title[:50] # Truncated

    def show_progress_bar(self, current: int, total: int, bar_length: int = 40):
        """Displays a progress bar in the console."""
        if total <= 0: return
        percent = float(current) / total
        arrow_length = max(0, int(round(percent * bar_length) - 1))
        arrow = '=' * arrow_length + '>'
        spaces = ' ' * (bar_length - len(arrow))
        percent_display = min(100, int(percent * 100)) # Cap at 100%
        # Simple progress display without language config dependency
        sys.stdout.write(f"\rProgress: [{arrow}{spaces}] {percent_display}% ({current}/{total})")
        sys.stdout.flush()
        if current >= total:
             sys.stdout.write('\n') # New line at the end

    def retry_api_call(self, call_function, *args, **kwargs):
        """Performs an API call with automatic retry on overload errors."""
        retries = 0
        # Use German for generic retry messages, or English fallback
        lang_conf_retry = self.LANGUAGE_CONFIG.get("Deutsch", self.LANGUAGE_CONFIG.get("Englisch"))
        while retries <= self.MAX_RETRIES:
            try:
                return call_function(*args, **kwargs)
            except Exception as e:
                error_str = str(e).lower()
                is_retryable = ("overloaded" in error_str or "rate_limit" in error_str or
                                "timeout" in error_str or "503" in error_str or "504" in error_str)

                if is_retryable and retries < self.MAX_RETRIES:
                    retries += 1
                    current_retry_delay = (self.DEFAULT_RETRY_DELAY_S *
                                           (self.RETRY_BACKOFF_FACTOR ** (retries - 1)) +
                                           random.uniform(0.1, 0.5)) # Jitter
                    logging.warning(lang_conf_retry["ERROR_API_OVERLOAD"].format(
                        delay=current_retry_delay, retries=retries, max_retries=self.MAX_RETRIES))
                    time.sleep(current_retry_delay)
                else:
                    logging.error(lang_conf_retry["ERROR_API_CALL_FAILED"].format(error=str(e)))
                    raise # Re-raise the original error if not retryable or retries exhausted
        # This part should ideally not be reached if the loop logic is correct,
        # but raise an exception just in case.
        raise Exception(lang_conf_retry["ERROR_ALL_RETRIES_FAILED"])

    def _create_system_prompt(self, language: str, template_key: str, context: Dict[str, Any]) -> str:
        """Creates a system prompt based on the language and a template."""
        lang_conf = self._get_lang_config(language)
        template = lang_conf.get(template_key)
        if not template:
            raise ValueError(f"Prompt template '{template_key}' not found for language '{language}'.")

        context['sprache'] = language # Add language to context
        context.setdefault('zusatz_anweisungen', '') # Use the same key for simplicity
        if context['zusatz_anweisungen']:
             # Format additional instructions
             # Determine the header based on language
             additional_header = "Zusätzliche Anweisungen" if language == "Deutsch" else "Additional Instructions"
             context['zusatz_anweisungen'] = f"\n**{additional_header}:**\n{context['zusatz_anweisungen']}\n"
        else:
             context['zusatz_anweisungen'] = "" # Ensure it's an empty string if None

        # Ensure all required keys for the specific template are present or provide defaults
        # This prevents KeyError during .format() if context is incomplete
        # Note: Keys like 'wortanzahl', 'titel' remain the same across languages for .format()
        if template_key == "PROMPT_STORY_GEN":
             context.setdefault('wortanzahl', self.DEFAULT_TARGET_WORDS)
             context.setdefault('titel', 'Untitled' if language == "Englisch" else 'Unbenannt')
             context.setdefault('prompt', '')
             context.setdefault('setting', '')
        elif template_key == "PROMPT_OUTLINE_GEN":
             context.setdefault('kapitel_anzahl', 0)
             context.setdefault('titel', 'Untitled' if language == "Englisch" else 'Unbenannt')
             context.setdefault('prompt', '')
             context.setdefault('setting', '')
             context.setdefault('wortanzahl', self.DEFAULT_TARGET_WORDS)
        elif template_key == "PROMPT_CHAPTER_GEN":
             context.setdefault('kapitel_nummer', 0)
             context.setdefault('kapitel_anzahl', 0)
             context.setdefault('titel', 'Untitled' if language == "Englisch" else 'Unbenannt')
             context.setdefault('prompt', '')
             context.setdefault('setting', '')
             context.setdefault('plot_outline', '[No Outline]' if language == "Englisch" else '[Keine Outline]')
             context.setdefault('zusammenfassung_vorher', '')
             context.setdefault('kapitel_wortanzahl', 1000)
             context.setdefault('min_kapitel_worte', 800)
             context.setdefault('max_kapitel_worte', 1500)
        elif template_key == "PROMPT_EPILOG_GEN":
             context.setdefault('titel', 'Untitled' if language == "Englisch" else 'Unbenannt')
             context.setdefault('plot_outline', '[No Outline]' if language == "Englisch" else '[Keine Outline]')
             context.setdefault('zusammenfassung_vorher', '')
             context.setdefault('letztes_kapitel_ende', '')

        try:
            return template.format(**context)
        except KeyError as e:
            logging.error(f"Missing key for prompt template '{template_key}' in language '{language}': {e}")
            logging.error(f"Context provided: {context}")
            raise ValueError(f"Failed to format prompt template '{template_key}' due to missing key: {e}")


    def generate_story(self, prompt: str, setting: str, title: str,
                       word_count: int = DEFAULT_TARGET_WORDS,
                       language: str = "Deutsch",
                       additional_instructions: Optional[str] = None,
                       chapter_mode: bool = True,
                       max_words_per_chapter: int = MAX_WORDS_PER_CHAPTER
                       ) -> Optional[str]:
        """Generates a short story with a focus on quality."""
        lang_conf = self._get_lang_config(language)
        logging.info(lang_conf["INFO_GENERATING_STORY"].format(
            titel=title, wortanzahl=word_count, sprache=language # Use consistent key names
        ))

        # Use the English variable names now for consistency
        if word_count < self.MIN_STORY_WORDS:
            logging.warning(lang_conf["WARN_LOW_WORDCOUNT"].format(
                wortanzahl=word_count, min_val=self.MIN_STORY_WORDS))
            word_count = self.MIN_STORY_WORDS
        elif word_count > self.MAX_STORY_WORDS_NO_CHAPTERS and not chapter_mode:
            logging.warning(lang_conf["WARN_HIGH_WORDCOUNT"].format(
                wortanzahl=word_count, max_val=self.MAX_STORY_WORDS_NO_CHAPTERS))
            word_count = self.MAX_STORY_WORDS_NO_CHAPTERS

        # Check if chapter mode should be activated
        effective_max_words = max_words_per_chapter # Default
        activate_chapters = (chapter_mode and word_count > effective_max_words)

        if activate_chapters:
             logging.info(lang_conf["INFO_CHAPTER_MODE_ACTIVATED"].format(wortanzahl=word_count))
             return self._generate_story_with_chapters(
                 prompt, setting, title, word_count, language,
                 additional_instructions, max_words_per_chapter
             )
        else:
            # If not using chapter mode, calculate buffer differently if needed
            # word_count_adjusted = int(word_count * self.WORD_COUNT_BUFFER_FACTOR) if not activate_chapters else word_count
            # logging.info(f"Adjusted target word count (with buffer {self.WORD_COUNT_BUFFER_FACTOR:.1f}x): {word_count_adjusted}")

            return self._generate_single_story(
                prompt, setting, title, word_count, language, additional_instructions # Pass original word_count here
            )

    def _generate_single_story(self, prompt: str, setting: str, title: str,
                               word_count: int, language: str,
                               additional_instructions: Optional[str]
                               ) -> Optional[str]:
        """Generates a shorter story in a single API call with quality focus."""
        lang_conf = self._get_lang_config(language)
        safe_title = self._safe_filename(title)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Use tempfile for safer temporary files
        temp_fd, temp_filename = tempfile.mkstemp(prefix=f"temp_story_{safe_title}_{timestamp}_", suffix=".txt")
        os.close(temp_fd) # Close handle, we just need the name

        try:
            # Use the REVISED prompt
            system_prompt = self._create_system_prompt(
                language, "PROMPT_STORY_GEN", {
                    "wortanzahl": word_count, # Use 'wortanzahl' as key expected by prompt template
                    "titel": title,
                    "prompt": prompt,
                    "setting": setting,
                    "zusatz_anweisungen": additional_instructions # Use 'zusatz_anweisungen' key
                })

            # Apply buffer factor *here* for token calculation if desired, or rely on ratio
            # Adjusted target for token calculation could be word_count * self.WORD_COUNT_BUFFER_FACTOR
            max_tokens = min(int(word_count * self.TOKEN_WORD_RATIO * 1.1), self.MAX_TOKENS_PER_CALL) # Small buffer added
            temperature = 0.75 # Possibly slightly lower for more coherence despite quality instructions
            logging.info(lang_conf["INFO_SENDING_API_REQUEST"].format(max_tokens=max_tokens, temp=temperature))

            start_time = time.time()
            logging.info(lang_conf["INFO_WAITING_API_RESPONSE"])

            # Use the specific user prompt text for the language
            user_prompt_text = lang_conf["USER_PROMPT_STORY"].format(titel=title, wortanzahl=word_count)

            response = self.retry_api_call(
                self.client.chat.completions.create,
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt_text},
                ]
            )

            story_content = response.choices[0].message.content

            duration = time.time() - start_time
            logging.info(lang_conf["INFO_GENERATION_COMPLETE"].format(dauer=duration)) # 'dauer' key used in format string

            # Save to temp file (for debugging or recovery)
            try:
                with open(temp_filename, 'w', encoding='utf-8') as f:
                    f.write(story_content)
                logging.debug(lang_conf["INFO_SAVED_TEMP_FILE"].format(dateiname=temp_filename)) # 'dateiname' key
            except Exception as e:
                logging.warning(lang_conf["WARN_CANNOT_WRITE_TEMP"].format(error=str(e)))

            # Format and clean the generated story
            story_content = self._format_story(story_content, title, language)
            story_content = self._clean_text_ending(story_content, language)

            return story_content

        except Exception as e:
            logging.error(f"Error during API request for single story: {str(e)}")
            # Attempt recovery from temp file
            if os.path.exists(temp_filename):
                try:
                    with open(temp_filename, 'r', encoding='utf-8') as f:
                        partial_story = f.read()
                    if len(partial_story) > self.MIN_CHARS_FOR_RESCUE:
                        logging.info(lang_conf["INFO_RESCUED_PARTIAL_CONTENT"].format(chars=len(partial_story)))
                        partial_story = self._format_story(partial_story, title, language)
                        partial_story = self._clean_text_ending(partial_story, language)
                        return partial_story
                except Exception as e2:
                    logging.error(lang_conf["ERROR_READING_TEMP_FILE"].format(error=str(e2)))
            return None # Generation failed
        finally:
            # Clean up temp file
            if os.path.exists(temp_filename):
                try:
                    os.remove(temp_filename)
                except Exception as e:
                    logging.warning(lang_conf["WARN_CANNOT_DELETE_TEMP"].format(error=str(e)))

    def _format_story(self, story_content: str, title: str, language: str) -> str:
        """Ensures the story starts correctly with the title."""
        story_content = story_content.strip()
        title_prefix = f"# {title}"
        if not story_content.startswith(title_prefix):
            # Try to remove any title generated by the model itself
            lines = story_content.split('\n')
            # Remove all lines at the beginning starting with #, except the expected title
            while lines and lines[0].strip().startswith("#") and lines[0].strip() != title_prefix:
                lines.pop(0)
            story_content = "\n".join(lines).strip()
            # Add the correct title if it's still missing
            if not story_content.startswith(title_prefix):
                 story_content = f"{title_prefix}\n\n{story_content}"
        return story_content

    def _clean_text_ending(self, text: str, language: str) -> str:
        """Checks and corrects abrupt endings in the text."""
        # This function can be kept largely as is, relying on regex and basic string ops.
        # It focuses on technical cleanup, not style.
        if not text or len(text) < self.MIN_CHARS_FOR_RESCUE // 2:
            return text

        lang_conf = self._get_lang_config(language)
        original_text = text
        text = text.rstrip()

        # 1. Incomplete sentence at the end (a... Z) - improved regex
        # Looks for a lowercase word, optional comma/semicolon, whitespace, then an uppercase word at the very end.
        match = re.search(r'([a-zäöüß]+[,;]?)\s+([A-ZÄÖÜ][a-zäöüß]*)\s*$', text)
        if match:
            # Find the last sentence punctuation before the potential incomplete part
            last_sentence_end = -1
            for punc in ['. ', '! ', '? ', '." ', '!" ', '?" ']:
                last_sentence_end = max(last_sentence_end, text.rfind(punc, 0, match.start()))

            if last_sentence_end > 0:
                # Calculate the actual end index after the punctuation
                end_index = last_sentence_end + 1 # After the punctuation mark itself
                # Check for space and quote after punctuation (e.g., ". ")
                if text[last_sentence_end+1:last_sentence_end+2] in [' "', ' !', ' ?']: end_index += 1
                text = text[:end_index].strip() # Keep the space after punctuation if it was there
                logging.info(lang_conf["INFO_REMOVED_INCOMPLETE_SENTENCE"])
            else:
                # If no prior sentence found, this might be the *only* sentence and it's incomplete.
                # Harder to fix reliably, maybe just log it.
                logging.debug("Could not reliably fix potentially incomplete single sentence.")


        # 2. No sentence-ending punctuation
        ends_with_punctuation = any(text.endswith(p) for p in ['.', '!', '?', '."', '!"', '?"'])
        last_char_is_letter = text[-1].isalpha() if text else False

        if not ends_with_punctuation and last_char_is_letter:
             # Find the last punctuation mark anywhere before the end
             last_sentence_end = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
             if last_sentence_end > 0:
                 # Truncate after the last found punctuation
                 text = text[:last_sentence_end + 1]
                 logging.info(lang_conf["INFO_CORRECTING_ENDING"] + " (missing punctuation)")
             else:
                 # No punctuation found at all? Very unlikely for long text.
                 # Maybe add a period as a last resort? Or leave it? For now, leave it.
                 logging.debug("Text ends without punctuation, and no prior punctuation found.")


        # 3. Unclosed quotation marks in the last paragraph
        paragraphs = text.split('\n\n')
        last_paragraph = paragraphs[-1].strip() if paragraphs else ""
        if last_paragraph:
            quote_count = last_paragraph.count('"')
            if quote_count % 2 != 0: # Odd number of quotes means one is likely unclosed
                last_quote_index_in_para = last_paragraph.rfind('"')
                # Find the last sentence end *within the paragraph*
                last_sentence_end_in_para = max(last_paragraph.rfind('.'), last_paragraph.rfind('!'), last_paragraph.rfind('?'))

                # If the last quote is after the last sentence end, it's likely part of incomplete dialogue
                if last_quote_index_in_para != -1 and last_quote_index_in_para > last_sentence_end_in_para:
                    # Try to cut off after the last complete sentence before the quote
                    text_before = '\n\n'.join(paragraphs[:-1])
                    if text_before: text_before += '\n\n' # Re-add paragraph break if needed

                    cutoff_point_in_para = max(last_paragraph.rfind(p, 0, last_quote_index_in_para) for p in ['.', '!', '?'])
                    if cutoff_point_in_para >= 0 : # Found punctuation before the dangling quote
                        text = text_before + last_paragraph[:cutoff_point_in_para + 1]
                        logging.info(lang_conf["INFO_REMOVED_INCOMPLETE_DIALOG"])
                    else:
                        # No punctuation before the quote in this paragraph, maybe remove the whole para? Risky.
                        # Fallback: Add a closing quote if the text doesn't already end with one
                        if not text.endswith('"'):
                            text += '"'
                            logging.info(lang_conf["INFO_ADDED_MISSING_QUOTE"] + " (fallback, unmatched quote)")

                elif not text.endswith('"'):
                    # Quote count is odd, but the last quote isn't dangling at the end.
                    # Most likely scenario: Forgot to close the last quote mark.
                    text += '"'
                    logging.info(lang_conf["INFO_ADDED_MISSING_QUOTE"])


        # 4. Short last paragraph or ends with a conjunction
        paragraphs = text.split('\n\n') # Recalculate in case text changed
        if len(paragraphs) > 1:
            last_paragraph_words = paragraphs[-1].strip().split()
            # Get language-specific conjunctions
            conjunctions = lang_conf.get("CONJUNCTIONS_AT_END", [])
            ends_with_conjunction = False
            if last_paragraph_words:
                 last_word_cleaned = last_paragraph_words[-1].lower().strip('".!?,')
                 ends_with_conjunction = last_word_cleaned in conjunctions

            if len(last_paragraph_words) < self.MIN_WORDS_FOR_VALID_ENDING or ends_with_conjunction:
                # Remove the last paragraph
                text = '\n\n'.join(paragraphs[:-1]).strip()
                logging.info(lang_conf["INFO_REMOVED_INCOMPLETE_PARAGRAPH"])

        # Log if changes were made
        if text != original_text:
             logging.debug(f"Text ending cleaned. Before: ...{original_text[-50:]}, After: ...{text[-50:]}")
        return text

    # --- Chapter structure, name/sentence extraction (can remain as is, logic is language-agnostic enough) ---
    def _optimize_chapter_structure(self, total_word_count: int, max_words_per_chapter: int) -> Tuple[int, List[int]]:
        """Calculates a simple chapter structure."""
        num_chapters = 0
        # If the total count is only slightly above the max per chapter, just make 1 or 2 chapters
        if total_word_count <= max_words_per_chapter * 1.5:
            num_chapters = math.ceil(total_word_count / max_words_per_chapter)
        else:
            # Otherwise, aim for a certain number of chapters based on total length, or minimum required
            num_chapters = max(self.MIN_CHAPTERS_LONG_STORY,
                               round(total_word_count / self.TARGET_WORDS_PER_CHAPTER_DIVISOR))

        # Ensure num_chapters is at least 1
        num_chapters = max(1, num_chapters)

        base_words = total_word_count // num_chapters
        remainder = total_word_count % num_chapters
        words_per_chapter = [base_words] * num_chapters
        # Distribute remainder words to the first few chapters
        for i in range(remainder):
            words_per_chapter[i] += 1

        # Recalculate if any chapter significantly exceeds the max (simple recalculation)
        if any(w > max_words_per_chapter * 1.2 for w in words_per_chapter): # Allow slight overshoot
            logging.debug(f"Recalculating chapter structure as max words ({max_words_per_chapter}) was exceeded.")
            new_num_chapters = num_chapters + 1 # Just add one more chapter
            base_words = total_word_count // new_num_chapters
            remainder = total_word_count % new_num_chapters
            words_per_chapter = [base_words] * new_num_chapters
            for i in range(remainder): words_per_chapter[i] += 1
            # Check again, log warning if still problematic
            if any(w > max_words_per_chapter * 1.2 for w in words_per_chapter):
                 logging.warning(f"Chapter structure could not strictly adhere to max {max_words_per_chapter} words per chapter.")

        return num_chapters, words_per_chapter

    def _extract_character_names(self, text: str, language: str) -> List[str]:
        """Extracts potential character names (heuristic)."""
        lang_conf = self._get_lang_config(language)
        names = set()
        # Regex for capitalized words (not directly after sentence end)
        # Positive lookbehind assertion: (?<!...) ensures the pattern is NOT preceded by ". ", "! ", or "? "
        potential_names = re.findall(r'(?<!\.\s)(?<!\?\s)(?<!\!\s)\b([A-ZÄÖÜ][a-zäöüß]+)\b', text)
        common_prefixes = lang_conf.get("COMMON_NOUN_PREFIXES", []) # e.g., "The", "Der", "Die", "Das"
        for name in potential_names:
            # Filter out common prefixes and very short words
            if name not in common_prefixes and len(name) > 2:
                names.add(name)
        return list(names)

    def _extract_important_sentences(self, text: str, max_sentences: int, language: str) -> List[str]:
        """Extracts the most important sentences (heuristic)."""
        lang_conf = self._get_lang_config(language)
        # Split into sentences - replace newlines with spaces first for better splitting
        # Positive lookbehind `(?<=...)`: matches if the preceding text ends with '.', '!', or '?'
        # Positive lookahead `(?=...)`: matches if the following text starts with whitespace and an uppercase letter
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-ZÄÖÜ])', text.replace('\n',' '))
        if not sentences: return []

        sentence_scores = []
        action_verbs = lang_conf.get("ACTION_VERBS", [])
        emotional_words = lang_conf.get("EMOTIONAL_WORDS", [])
        character_names = self._extract_character_names(text, language) # Extract once

        for i, sentence in enumerate(sentences):
            score = 0
            sentence_lower = sentence.lower()
            word_count = len(sentence.split())

            # Scoring criteria (adjust weights as needed)
            if i < 3 or i >= len(sentences) - 3: score += 3 # Position (beginning/end)
            if 10 < word_count < 35: score += 2 # Optimal length
            elif word_count >= 35: score += 1 # Longer sentences might be relevant
            if any(name in sentence for name in character_names): score += 4 # Contains character name
            if any(f' {verb} ' in sentence_lower for verb in action_verbs): score += 2 # Contains action verb
            if any(emotion in sentence_lower for emotion in emotional_words): score += 2 # Contains emotional word
            if '"' in sentence: score += 1 # Contains dialogue

            sentence_scores.append((sentence.strip(), score, i)) # Store sentence, score, original index

        # Sort by score (descending), then by original index (ascending) to maintain order among ties
        important_sentences_data = sorted(sentence_scores, key=lambda x: (-x[1], x[2]))[:max_sentences]

        # Sort the selected sentences back by their original index to preserve flow
        top_sentences_final = sorted(important_sentences_data, key=lambda x: x[2])

        return [sentence_data[0] for sentence_data in top_sentences_final]

    def _create_chapter_summary(self, chapter_texts: List[str], current_chapter_index: int, language: str) -> str:
        """Automatically creates a summary of previous chapters (heuristic)."""
        # current_chapter_index is 0-based index of the chapter *about to be generated*
        lang_conf = self._get_lang_config(language)
        if current_chapter_index <= 0 or not chapter_texts:
            return lang_conf["SUMMARY_FIRST_CHAPTER"] # No previous chapters

        # Limit summary to the last few chapters to keep context relevant and prompt short
        max_summary_chapters = 3
        start_index = max(0, current_chapter_index - max_summary_chapters)
        relevant_previous_chapters = chapter_texts[start_index:current_chapter_index] # Slice up to current index

        summary_parts = []

        for i, chapter_text in enumerate(relevant_previous_chapters):
            actual_chapter_number = start_index + i + 1 # 1-based chapter number
            # Extract more sentences from the immediately preceding chapter
            num_sentences = 5 if actual_chapter_number == current_chapter_index else 3
            important_sentences = self._extract_important_sentences(chapter_text, num_sentences, language)

            if important_sentences:
                 # Choose the correct prefix ("Previously in..." or "Chapter X:")
                 prefix_key = "SUMMARY_LAST_CHAPTER_PREFIX" if actual_chapter_number == current_chapter_index else "SUMMARY_CHAPTER_PREFIX"
                 prefix = lang_conf[prefix_key].format(num=actual_chapter_number)
                 summary_parts.append(f"{prefix} {' '.join(important_sentences)}")

        if not summary_parts:
            # Fallback if no important sentences could be extracted
            return lang_conf["SUMMARY_FIRST_CHAPTER"]

        # Add introduction and join parts
        intro = lang_conf['SUMMARY_PREVIOUS_CHAPTERS_INTRO']
        return f"\n\n{intro}\n" + "\n".join(summary_parts)


    # --- Generation with Chapters (Uses revised prompts) ---
    def _generate_story_with_chapters(self, prompt: str, setting: str, title: str,
                                      total_word_count: int, language: str,
                                      additional_instructions: Optional[str],
                                      max_words_per_chapter: int
                                      ) -> Optional[str]:
        """Generates a longer story by dividing it into chapters with quality focus."""
        lang_conf = self._get_lang_config(language)
        safe_title = self._safe_filename(title)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. Chapter Structure
        num_chapters, words_per_chapter_list = self._optimize_chapter_structure(total_word_count, max_words_per_chapter)
        logging.info(lang_conf["INFO_GENERATING_CHAPTERS"].format(kapitel_anzahl=num_chapters)) # Use key from lang_conf
        logging.debug(f"Planned word distribution: {words_per_chapter_list}")

        # 2. Generate Plot Outline
        logging.info(lang_conf["INFO_GENERATING_OUTLINE"])
        plot_outline = None
        # Determine output directory for the outline
        # If run from main_app.py, args might not be directly accessible here.
        # Rely on a default or potentially pass output_dir down. For now, save relative.
        # A better approach might be to return the outline path from this method.
        output_dir_outline = "." # Default to current directory
        # Check if args exist (might be set if run standalone)
        global args # Access global args if available (less ideal, but works for standalone)
        if 'args' in globals() and hasattr(args, 'output_dir'):
             output_dir_outline = args.output_dir

        outline_final_filename = os.path.join(output_dir_outline, f"outline_{safe_title}_{timestamp}.txt")
        temp_outline_fd, temp_outline_path = tempfile.mkstemp(suffix=".txt", prefix="outline_")
        os.close(temp_outline_fd) # Close handle

        try:
            # Create the system prompt for the outline
            outline_prompt_text = self._create_system_prompt(
                language, "PROMPT_OUTLINE_GEN", {
                    "kapitel_anzahl": num_chapters, # Key expected by template
                    "titel": title,
                    "prompt": prompt,
                    "setting": setting,
                    "wortanzahl": total_word_count, # Key expected by template
                    "zusatz_anweisungen": additional_instructions # Key expected by template
                })
            # User prompt for the outline request
            user_prompt_outline = lang_conf["USER_PROMPT_OUTLINE"].format(kapitel_anzahl=num_chapters, titel=title)
            logging.info(lang_conf["INFO_WAITING_API_RESPONSE"] + " (Outline)")

            outline_response = self.retry_api_call(
                self.client.chat.completions.create,
                model=self.model_name,
                max_tokens=min(6000, self.MAX_TOKENS_PER_CALL), # Generous token limit for outline
                temperature=0.7, # Temperature for outline can be slightly higher for creativity
                messages=[ {"role": "system", "content": outline_prompt_text}, {"role": "user", "content": user_prompt_outline} ]
            )
            plot_outline = outline_response.choices[0].message.content
            logging.info(lang_conf["INFO_OUTLINE_CREATED"])

            # Save outline (temporary for debug, final for user)
            try:
                # Save to temp file first
                with open(temp_outline_path, 'w', encoding='utf-8') as f_temp:
                     f_temp.write(plot_outline)
                logging.debug(lang_conf["INFO_SAVED_TEMP_FILE"].format(dateiname=temp_outline_path)) # 'dateiname' key

                # Attempt to save the final outline file
                os.makedirs(os.path.dirname(outline_final_filename), exist_ok=True)
                with open(outline_final_filename, 'w', encoding='utf-8') as f_final:
                     outline_header = f"Plot Outline for '{title}'" if language == "Englisch" else f"Plot-Outline für '{title}'"
                     f_final.write(f"# {outline_header}\n\n{plot_outline}")
                logging.info(lang_conf["INFO_SAVED_OUTLINE"].format(dateiname=outline_final_filename)) # 'dateiname' key
            except Exception as e:
                 logging.warning(lang_conf["WARN_CANNOT_SAVE_OUTLINE"].format(error=str(e)))
                 # Fallback save logic could be added here if crucial

        except Exception as e:
             logging.error(f"Error creating plot outline: {str(e)}. Continuing generation without it.")
             plot_outline = "[Outline could not be generated]" if language == "Englisch" else "[Outline konnte nicht generiert werden]"
        finally:
             # Delete temporary outline file
             if os.path.exists(temp_outline_path):
                 try: os.remove(temp_outline_path)
                 except Exception as e: logging.warning(lang_conf["WARN_CANNOT_DELETE_TEMP"].format(error=str(e)))


        # 3. Generate Chapters Individually
        generated_chapters: List[str] = []
        previous_summary = "" # Start with empty summary for the first chapter call
        self.show_progress_bar(0, num_chapters)

        for i in range(num_chapters):
            chapter_number = i + 1
            target_words_chapter = words_per_chapter_list[i]
            logging.info(lang_conf["INFO_GENERATING_CHAPTER_NUM"].format(
                kapitel_nummer=chapter_number, kapitel_anzahl=num_chapters # Keys from lang_conf
            ))

            # Create summary of *previous* chapters for context
            # Pass the 0-based index `i`
            if i > 0:
                 previous_summary = self._create_chapter_summary(generated_chapters, i, language)
            else:
                 previous_summary = lang_conf["SUMMARY_FIRST_CHAPTER"] # For the very first chapter prompt

            # Context for the chapter prompt
            chapter_context = {
                "kapitel_nummer": chapter_number, # Key expected by template
                "kapitel_anzahl": num_chapters,   # Key expected by template
                "titel": title,
                "prompt": prompt,
                "setting": setting,
                "plot_outline": plot_outline, # Pass the generated or fallback outline
                "zusammenfassung_vorher": previous_summary, # Key expected by template
                "kapitel_wortanzahl": target_words_chapter, # Key expected by template
                "min_kapitel_worte": int(target_words_chapter * 0.8), # Key expected by template
                "max_kapitel_worte": int(target_words_chapter * 1.5), # Key expected by template, allow larger range
                "zusatz_anweisungen": additional_instructions # Key expected by template
            }

            try:
                # Call the API for a single chapter
                chapter_text = self._generate_single_chapter_api(
                    chapter_number, num_chapters, chapter_context, language, title
                )
                generated_chapters.append(chapter_text)

            except Exception as e:
                # Log error and add a placeholder chapter
                logging.error(lang_conf["ERROR_API_REQUEST_CHAPTER"].format(
                    kapitel_nummer=chapter_number, error=str(e))) # Keys from lang_conf
                error_title_key = "ERROR_CHAPTER_TITLE"
                error_content_key = "ERROR_CHAPTER_CONTENT"
                error_title = lang_conf[error_title_key].format(kapitel_nummer=chapter_number) # Key from lang_conf
                error_content = lang_conf[error_content_key]
                chapter_text = f"# {error_title}\n\n{error_content}"
                # Add main title only for the first chapter if it fails
                if chapter_number == 1:
                    chapter_text = f"# {title}\n\n{chapter_text}"
                generated_chapters.append(chapter_text)

            self.show_progress_bar(chapter_number, num_chapters)


        # 4. Combine Chapters
        full_story = self._combine_chapters(generated_chapters, title, language)


        # 5. Generate Epilogue if the last chapter seems incomplete
        if generated_chapters and num_chapters > 0:
            last_chapter_text = generated_chapters[-1]
            target_words_last_chapter = words_per_chapter_list[-1]
            actual_words_last = len(last_chapter_text.split())
            # Check if the last chapter is an error placeholder
            is_error_chapter = lang_conf["ERROR_CHAPTER_TITLE"].format(kapitel_nummer=num_chapters) in last_chapter_text

            # Condition to generate epilogue: last chapter not an error and significantly shorter than planned
            if not is_error_chapter and actual_words_last < target_words_last_chapter * 0.7:
                logging.info(lang_conf["INFO_LAST_CHAPTER_INCOMPLETE"])
                # Need the summary *up to* the last chapter
                summary_before_epilogue = self._create_chapter_summary(generated_chapters, num_chapters, language)

                epilogue_text = self._generate_epilogue(
                    title,
                    plot_outline, # Use the generated or fallback outline
                    summary_before_epilogue,
                    last_chapter_text, # Pass the potentially incomplete text
                    language,
                    additional_instructions
                )
                if epilogue_text:
                    # Append epilogue cleanly, avoid double newlines
                    full_story = full_story.strip() + "\n\n" + epilogue_text.strip()

        # Final cleanup (minimal)
        full_story = self._clean_text_ending(full_story.strip(), language)

        return full_story


    def _generate_single_chapter_api(self, chapter_number: int, total_chapters: int,
                                     chapter_context: Dict[str, Any],
                                     language: str, title: str
                                     ) -> str:
        """Generates a single chapter via the API with quality focus."""
        lang_conf = self._get_lang_config(language)
        # Temporary file for this chapter (optional for debugging)
        # safe_title = self._safe_filename(title)
        # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # chapter_temp_file = f"temp_chapter_{chapter_number}_{safe_title}_{timestamp}.txt"
        temp_chapter_fd, temp_chapter_path = tempfile.mkstemp(suffix=".txt", prefix=f"chapter_{chapter_number}_")
        os.close(temp_chapter_fd) # Close handle

        try:
            target_words = chapter_context["kapitel_wortanzahl"] # Key from context dict
            # Token buffer to allow space for quality instructions and potentially longer output
            max_tokens = min(int(target_words * self.TOKEN_WORD_RATIO * 1.1) + 150, self.MAX_TOKENS_PER_CALL)
            temperature = 0.75 # Adjusted temperature

            # Use the REVISED prompt for chapters
            system_prompt = self._create_system_prompt(language, "PROMPT_CHAPTER_GEN", chapter_context)
            # User prompt for the chapter request
            user_prompt = lang_conf["USER_PROMPT_CHAPTER"].format(kapitel_nummer=chapter_number, titel=title) # Keys from lang_conf

            logging.info(lang_conf["INFO_WAITING_API_RESPONSE"] + f" (Chapter {chapter_number})")
            start_time = time.time()

            response = self.retry_api_call(
                self.client.chat.completions.create,
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[ {"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt} ]
            )

            chapter_text = response.choices[0].message.content
            duration = time.time() - start_time
            logging.info(lang_conf["INFO_CHAPTER_COMPLETE"].format(kapitel_nummer=chapter_number, dauer=duration)) # Keys from lang_conf

            # Save temporarily (optional)
            try:
                with open(temp_chapter_path, 'w', encoding='utf-8') as f: f.write(chapter_text)
                logging.debug(lang_conf["INFO_SAVED_TEMP_FILE"].format(dateiname=temp_chapter_path)) # 'dateiname' key
            except Exception as e: logging.warning(lang_conf["WARN_CANNOT_WRITE_TEMP"].format(error=str(e)))

            # Validation (minimum length check)
            if len(chapter_text.split()) < 50:
                 logging.warning(f"Chapter {chapter_number} is very short ({len(chapter_text.split())} words).")
                 # Do not throw an error here, just log it.

            # Formatting and cleanup
            chapter_text = self._format_chapter(chapter_number, chapter_text, title, language)
            chapter_text = self._clean_text_ending(chapter_text, language)

            return chapter_text

        except Exception as e:
            logging.error(f"Error in API call for chapter {chapter_number}: {e}")
            # Attempt rescue from temp file
            if os.path.exists(temp_chapter_path):
                try:
                    with open(temp_chapter_path, 'r', encoding='utf-8') as f: partial_chapter = f.read()
                    if len(partial_chapter) > self.MIN_CHARS_FOR_RESCUE:
                        logging.info(lang_conf["INFO_RESCUED_PARTIAL_CONTENT"].format(chars=len(partial_chapter)))
                        partial_chapter = self._format_chapter(chapter_number, partial_chapter, title, language)
                        partial_chapter = self._clean_text_ending(partial_chapter, language)
                        # Add notice about the issue
                        rescue_notice = lang_conf['RESCUED_CHAPTER_NOTICE']
                        return partial_chapter + f"\n\n{rescue_notice}"
                except Exception as e2: logging.error(lang_conf["ERROR_READING_TEMP_FILE"].format(error=str(e2)))

            # If rescue fails, re-raise the exception to signal failure in the main process
            raise e
        finally:
            # Delete temporary chapter file
            if os.path.exists(temp_chapter_path):
                try: os.remove(temp_chapter_path)
                except Exception as e: logging.warning(lang_conf["WARN_CANNOT_DELETE_TEMP"].format(error=str(e)))


    def _format_chapter(self, chapter_number: int, chapter_text: str, title: str, language: str) -> str:
        """Ensures the chapter is correctly formatted."""
        # This function's logic can remain largely the same.
        lang_conf = self._get_lang_config(language)
        chapter_text = chapter_text.strip()

        main_title_prefix = f"# {title}"
        # Define patterns and titles based on language
        if language == "Deutsch":
             # Allow optional colon and any text after chapter number
             chapter_prefix_pattern = rf"^\s*#\s*Kapitel\s+{chapter_number}\s*:?.*"
             default_chapter_title_template = lang_conf["DEFAULT_CHAPTER_TITLE"] # Template string
             first_chapter_title_template = lang_conf["FIRST_CHAPTER_TITLE"] # Template string
             default_chapter_title = default_chapter_title_template.format(kapitel_nummer=chapter_number)
             first_chapter_title = first_chapter_title_template # Assumes it doesn't need formatting now
        else: # English
             chapter_prefix_pattern = rf"^\s*#\s*Chapter\s+{chapter_number}\s*:?.*"
             default_chapter_title_template = lang_conf["DEFAULT_CHAPTER_TITLE"] # Template string
             first_chapter_title_template = lang_conf["FIRST_CHAPTER_TITLE"] # Template string
             default_chapter_title = default_chapter_title_template.format(kapitel_nummer=chapter_number)
             first_chapter_title = first_chapter_title_template # Assumes it doesn't need formatting

        # Check if the text already starts with a correct chapter heading
        has_correct_chapter_heading = bool(re.match(chapter_prefix_pattern, chapter_text, re.IGNORECASE))

        if chapter_number == 1:
            has_main_title = chapter_text.startswith(main_title_prefix)
            # Remove potentially incorrect main titles if they don't match the expected one
            lines = chapter_text.split('\n')
            while lines and lines[0].strip().startswith("#") and lines[0].strip() != main_title_prefix:
                lines.pop(0)
            chapter_text = "\n".join(lines).strip()

            # Add main title if missing
            if not chapter_text.startswith(main_title_prefix):
                 chapter_text = f"{main_title_prefix}\n\n{chapter_text}"

            # Add Chapter 1 heading if missing, placing it *after* the main title
            # Find the end of the main title line
            main_title_match = re.match(rf"^{re.escape(main_title_prefix)}\s*", chapter_text, re.IGNORECASE)
            if main_title_match:
                 # Check if chapter heading follows immediately (allow for newlines)
                 text_after_title = chapter_text[main_title_match.end():].lstrip('\n')
                 if not re.match(chapter_prefix_pattern, text_after_title, re.IGNORECASE):
                    # Insert chapter heading
                    insert_pos = main_title_match.end()
                    # Use the language-specific first chapter title
                    chapter_heading_text = first_chapter_title_template.format(kapitel_nummer=1) if "{kapitel_nummer}" in first_chapter_title_template else first_chapter_title

                    chapter_text = chapter_text[:insert_pos].rstrip() + f"\n\n# {chapter_heading_text}\n\n" + text_after_title
            else:
                # Should not happen if previous logic worked, but fallback just in case
                 chapter_heading_text = first_chapter_title_template.format(kapitel_nummer=1) if "{kapitel_nummer}" in first_chapter_title_template else first_chapter_title
                 chapter_text = f"{main_title_prefix}\n\n# {chapter_heading_text}\n\n{chapter_text}"

        elif not has_correct_chapter_heading:
            # Add default chapter heading if missing for chapters > 1
            # Use the language-specific default chapter title
            chapter_text = f"# {default_chapter_title}\n\n{chapter_text}"

        return chapter_text


    def _combine_chapters(self, chapter_texts: List[str], title: str, language: str) -> str:
        """Combines the generated chapter texts into a single story string."""
        lang_conf = self._get_lang_config(language)
        if not chapter_texts:
            error_msg = "[Error: No chapters generated]" if language == "Englisch" else "[Fehler: Keine Kapitel generiert]"
            return f"# {title}\n\n{error_msg}"

        # The first chapter text should already be correctly formatted (with main title and Chapter 1 title)
        full_story = chapter_texts[0].strip()

        # Append the rest of the chapters, ensuring proper separation
        for i in range(1, len(chapter_texts)):
            # Always add two newlines between chapters for paragraph separation
            full_story += "\n\n" + chapter_texts[i].strip()

        return full_story


    def _generate_epilogue(self, title: str, plot_outline: str, summary_before: str,
                           last_chapter_text: str, language: str,
                           additional_instructions: Optional[str]
                          ) -> Optional[str]:
        """Generates an epilogue with quality focus."""
        # This function primarily uses the revised epilogue prompt.
        lang_conf = self._get_lang_config(language)
        safe_title = self._safe_filename(title)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # epilogue_temp_file = f"temp_epilogue_{safe_title}_{timestamp}.txt"
        temp_epilog_fd, temp_epilog_path = tempfile.mkstemp(suffix=".txt", prefix="epilog_")
        os.close(temp_epilog_fd) # Close handle

        logging.info(lang_conf["INFO_GENERATING_EPILOG"])
        try:
            # Provide more context from the end of the last chapter
            last_chapter_ending_context = last_chapter_text[-700:]

            epilogue_context = {
                "titel": title,
                "plot_outline": plot_outline,
                "zusammenfassung_vorher": summary_before, # Summary *up to* the last chapter
                "letztes_kapitel_ende": last_chapter_ending_context, # Key expected by template
                "zusatz_anweisungen": additional_instructions # Key expected by template
            }
            # Use the REVISED Epilogue Prompt
            system_prompt = self._create_system_prompt(language, "PROMPT_EPILOG_GEN", epilogue_context)
            # User prompt for epilogue request
            user_prompt = lang_conf["USER_PROMPT_EPILOG"].format(titel=title) # Key from lang_conf

            logging.info(lang_conf["INFO_WAITING_API_RESPONSE"] + " (Epilogue)")
            start_time = time.time()

            response = self.retry_api_call(
                self.client.chat.completions.create,
                model=self.model_name,
                max_tokens=min(2500, self.MAX_TOKENS_PER_CALL), # More space for epilogue
                temperature=0.7, # Slightly higher temperature allowed for conclusion
                messages=[ {"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt} ]
            )
            epilogue_text = response.choices[0].message.content
            duration = time.time() - start_time
            logging.info(lang_conf["INFO_EPILOG_GENERATED"] + f" ({duration:.1f}s)")

            # Save temp file (optional)
            try:
                with open(temp_epilog_path, 'w', encoding='utf-8') as f: f.write(epilogue_text)
                logging.debug(lang_conf["INFO_SAVED_TEMP_FILE"].format(dateiname=temp_epilog_path)) # 'dateiname' key
            except Exception as e: logging.warning(lang_conf["WARN_CANNOT_WRITE_TEMP"].format(error=str(e)))

            # Format and clean epilogue
            epilogue_text = epilogue_text.strip()
            epilogue_title_text = lang_conf['EPILOG_TITLE'] # Get language-specific title
            epilogue_title_prefix = f"# {epilogue_title_text}"
            if not epilogue_text.startswith(epilogue_title_prefix):
                epilogue_text = f"{epilogue_title_prefix}\n\n{epilogue_text}"
            epilogue_text = self._clean_text_ending(epilogue_text, language)
            return epilogue_text

        except Exception as e:
            logging.error(lang_conf["ERROR_GENERATING_EPILOG"].format(error=str(e)))
            # Attempt rescue from temp file
            if os.path.exists(temp_epilog_path):
                try:
                    with open(temp_epilog_path, 'r', encoding='utf-8') as f: partial_epilogue = f.read()
                    if len(partial_epilogue) > self.MIN_CHARS_FOR_RESCUE:
                        logging.info(lang_conf["INFO_RESCUED_PARTIAL_CONTENT"].format(chars=len(partial_epilogue)))
                        partial_epilogue = self._clean_text_ending(partial_epilogue.strip(), language)
                        # Ensure title formatting
                        epilogue_title_text = lang_conf['EPILOG_TITLE']
                        epilogue_title_prefix = f"# {epilogue_title_text}"
                        if not partial_epilogue.startswith(epilogue_title_prefix):
                             partial_epilogue = f"{epilogue_title_prefix}\n\n{partial_epilogue}"
                        # Add rescue notice
                        rescue_notice = lang_conf['RESCUED_EPILOG_NOTICE']
                        return partial_epilogue + f"\n\n{rescue_notice}"
                except Exception as e2: logging.error(lang_conf["ERROR_READING_TEMP_FILE"].format(error=str(e2)))
            return None # Epilogue generation failed
        finally:
            # Delete temporary epilogue file
            if os.path.exists(temp_epilog_path):
                try: os.remove(temp_epilog_path)
                except Exception as e: logging.warning(lang_conf["WARN_CANNOT_DELETE_TEMP"].format(error=str(e)))


    def save_as_text_file(self, content: str, title: str, language: str, output_path: Optional[str] = None) -> str:
        """Saves the content as a text file."""
        # Logic can remain the same.
        lang_conf = self._get_lang_config(language)
        safe_title = self._safe_filename(title)
        filename = ""

        if output_path:
             # Check if output_path looks like a directory or a file
             path_isdir = os.path.isdir(output_path)
             path_has_extension = os.path.splitext(output_path)[1] != ""

             if path_isdir or not path_has_extension:
                  # Treat as directory: create filename within it
                  os.makedirs(output_path, exist_ok=True)
                  timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                  base_filename = f"story_{safe_title}_{timestamp}.txt" if language == "Englisch" else f"geschichte_{safe_title}_{timestamp}.txt"
                  filename = os.path.join(output_path, base_filename)
             else:
                  # Treat as a specific filename
                  filename = output_path
                  if not filename.lower().endswith(".txt"):
                       filename += ".txt"
                  # Ensure the directory for the specified file exists
                  os.makedirs(os.path.dirname(filename) or '.', exist_ok=True) # Use '.' if dirname is empty
        else:
            # Default: save in the current directory
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"story_{safe_title}_{timestamp}.txt" if language == "Englisch" else f"geschichte_{safe_title}_{timestamp}.txt"
            filename = base_filename

        # Format content (ensure title is at the start)
        formatted_content = self._format_story(content, title, language)

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                # Ensure there's a newline at the very end of the file
                f.write(formatted_content.strip() + "\n")
            logging.info(lang_conf["INFO_SAVED_TEXT_FILE"].format(dateiname=filename)) # 'dateiname' key
            return filename
        except Exception as e:
             logging.error(f"Error saving text file '{filename}': {str(e)}")
             raise # Re-raise the exception


# === Main Part / Command Line Interface ===
def main():
    parser = argparse.ArgumentParser(
        description="Generates short stories with a focus on quality",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--prompt", type=str, required=True, help="Prompt/Basic idea for the story")
    parser.add_argument("--setting", type=str, required=True, help="Setting (Place, Time, Atmosphere)")
    parser.add_argument("--title", type=str, required=True, help="Title of the story") # Renamed --titel
    parser.add_argument("--wordcount", type=int, default=ShortStoryGenerator.DEFAULT_TARGET_WORDS, # Renamed --wortanzahl
                        help="Approximate TARGET word count")
    parser.add_argument("--language", type=str, default="Deutsch", choices=ShortStoryGenerator.SUPPORTED_LANGUAGES, # Renamed --sprache
                        help="Language of the story")
    parser.add_argument("--additional", type=str, help="Additional instructions for the generation") # Renamed --zusatz
    parser.add_argument("--api-key", type=str, help="Nebius API Key (alternatively use NEBIUS_API_KEY env var)")
    parser.add_argument("--model", type=str, default=MODELL_NAME, help="Name of the LLM model to use")
    parser.add_argument("--save-text", action="store_true", help="Save the generated story as a text file")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Directory for output files (text story, outline)")
    # parser.add_argument("--author", type=str, default="AI Narrator", help="Author (currently not used)")
    parser.add_argument("--max-words-per-chapter", type=int, default=ShortStoryGenerator.MAX_WORDS_PER_CHAPTER, # Renamed
                        help="Maximum word count per chapter API call (triggers chapter mode)")
    parser.add_argument("--no-chapter-mode", action="store_true", # Renamed
                        help="Disable chapter splitting (not recommended for stories > max-words-per-chapter)")
    parser.add_argument("--debug", action="store_true", help="Enable detailed debug output")

    # Make args globally accessible for helper functions (like outline saving)
    # Note: Passing args explicitly is generally better practice.
    global args
    args = parser.parse_args()

    # Configure Logging Level
    log_level = logging.DEBUG if args.debug else logging.INFO
    # Force reconfiguration in case basicConfig was called before (e.g., by imports)
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
    logging.debug("Debug mode enabled." if args.debug else "Info mode enabled.")

    # Create output directory if it doesn't exist
    try:
        # Only create if it's not the current directory "."
        if args.output_dir and args.output_dir != ".":
             os.makedirs(args.output_dir, exist_ok=True)
             logging.info(f"Output directory ensured: {args.output_dir}")
    except Exception as e:
        logging.error(f"Error creating output directory '{args.output_dir}': {e}")
        sys.exit(1)

    # --- Instantiate Generator and Run ---
    generator = None # Initialize generator to None
    try:
        generator = ShortStoryGenerator(api_key=args.api_key, model=args.model)
        # Get language config based on user input AFTER generator is initialized
        lang_conf = generator._get_lang_config(args.language)

        # Apply word count buffer factor primarily for chapter mode calculation? Or always?
        # Let's apply it just before calling generate_story to decide if chapter mode is needed.
        # The generate_story method will handle the actual word count target for the prompts.
        # target_word_count_with_buffer = int(args.wordcount * ShortStoryGenerator.WORD_COUNT_BUFFER_FACTOR)
        # logging.info(f"Target word count for chapter mode decision (buffer {ShortStoryGenerator.WORD_COUNT_BUFFER_FACTOR:.1f}x): {target_word_count_with_buffer}")


        # Generate the story
        story = generator.generate_story(
            prompt=args.prompt,
            setting=args.setting,
            title=args.title,
            word_count=args.wordcount, # Pass the user's target word count
            language=args.language,
            additional_instructions=args.additional,
            chapter_mode=not args.no_chapter_mode,
            max_words_per_chapter=args.max_words_per_chapter
        )

        # Check if story generation was successful
        if not story:
            logging.error(lang_conf["ERROR_GENERATION_FAILED"])
            sys.exit(1)

        # Log final word count
        actual_word_count = len(story.split())
        logging.info(lang_conf["INFO_FINAL_WORD_COUNT"].format(wortanzahl=actual_word_count)) # Use key 'wortanzahl'

        # --- Output / Save ---
        if args.save_text:
            # Define the output path for the text file (can be dir or specific file)
            text_output_path = args.output_dir
            try:
                 # Save the generated story
                 generator.save_as_text_file(
                     content=story, title=args.title, language=args.language,
                     output_path=text_output_path
                 )
            except Exception as e:
                 logging.error(f"Could not save text file: {e}")
                 # Fallback: Print to console if saving fails
                 print("\n--- Story (Could not be saved) ---")
                 print(story)
                 print("--- End Story ---")
        else:
            # Print story to console
            print("\n" + "=" * 80)
            print(story.strip()) # Print only the content, title is already inside
            print("=" * 80)


    except ValueError as ve:
         # Catch configuration errors (e.g., unsupported language, missing API key)
         logging.error(f"Configuration Error: {ve}")
         sys.exit(1)
    except Exception as e:
        # Catch any other unexpected errors during execution
        # Try to get language config for the error message, default if necessary
        try:
             # Ensure generator was initialized before trying to get lang_conf
             if generator:
                 lang_conf = generator._get_lang_config(args.language)
             else:
                 # Attempt to load default language config if generator failed early
                 lang_conf = ShortStoryGenerator.LANGUAGE_CONFIG.get(args.language, ShortStoryGenerator.LANGUAGE_CONFIG['Deutsch'])
        except:
             # Ultimate fallback if everything fails
             lang_conf = ShortStoryGenerator.LANGUAGE_CONFIG['Deutsch'] # Or English

        logging.error(f"An unexpected error occurred: {str(e)}")
        if args.debug:
             import traceback
             traceback.print_exc() # Print full traceback in debug mode
        sys.exit(1)

if __name__ == "__main__":
    main()
# --- END OF FILE story_generator.py ---