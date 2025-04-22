# -*- coding: utf-8 -*-

import os
import sys
import argparse
# import json # Not currently used
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

load_dotenv() # Loads .env for API Key if not passed directly

# === Configuration and Constants (Adopted from Web Script) ===
MODELL_NAME = "microsoft/phi-4" # Or your preferred story model
API_BASE_URL = "https://api.studio.nebius.com/v1/"

# Retry Constants
DEFAULT_RETRY_DELAY_S: int = 15 # Longer delay for potentially longer tasks
MAX_RETRIES: int = 3
RETRY_BACKOFF_FACTOR: float = 1.5

# Word Count & Token Limits
MIN_STORY_WORDS: int = 500
MAX_STORY_WORDS_NO_CHAPTERS: int = 25000 # Theoretical limit for single call
DEFAULT_TARGET_WORDS: int = 5000
TOKEN_WORD_RATIO: float = 1.6 # Heuristic ratio
MAX_TOKENS_PER_CALL: int = 15000 # API Limit (Input + Output) - Check limits
MIN_CHAPTERS_LONG_STORY: int = 3
TARGET_WORDS_PER_CHAPTER_DIVISOR: int = 2500
WORD_COUNT_BUFFER_FACTOR: float = 2.8 # Buffer for LLM word count inaccuracy

# Text Cleaning Constants
MIN_WORDS_FOR_VALID_ENDING: int = 4
MIN_CHARS_FOR_RESCUE: int = 100

# --- Module-Level Constants ---
SUPPORTED_LANGUAGES: List[str] = ["Deutsch", "Englisch"]
MAX_WORDS_PER_CHAPTER: int = 3000 # Max words per chapter *generation call*

# Logging configuration (will be reconfigured in main based on args)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__) # Use standard logger

class StoryGenerator: # Renamed class
    """
    Generates short stories with quality guidelines in the prompt.
    Includes enhanced chapter context handling using LLM summaries (detailed & running),
    outline segmentation, and explicit hierarchical instructions for coherence.
    CLI Version: Includes file saving.
    """
    # --- Language configuration (Adopted from Web Script) ---
    LANGUAGE_CONFIG: Dict[str, Dict[str, Any]] = {
        "Deutsch": {
            # --- Story/Outline Prompts ---
            "PROMPT_STORY_GEN": """Du bist ein talentierter und erfahrener Autor von Kurzgeschichten in **deutscher Sprache**. Deine Aufgabe ist es, eine fesselnde, gut geschriebene Geschichte basierend auf den folgenden Vorgaben zu erstellen.

Ziel-Wortanzahl: ca. {wortanzahl} Wörter.

Titel: {titel}
Prompt (Grundidee): {prompt}
Setting (Ort, Zeit, Atmosphäre): {setting}

**Qualitätsrichtlinien für das Schreiben:**
-   **Stil & Fluss:** Schreibe in klarem, prägnantem und ansprechendem Deutsch. Variiere Satzlänge und -struktur, um Monotonie zu vermeiden. Sorge für einen natürlichen Lesefluss.
-   **Wiederholungen VERMEIDEN:** Achte SEHR GENAU darauf, Wort- und Phrasenwiederholungen sowie ähnliche Satzmuster zu minimieren. Nutze Synonyme oder formuliere Sätze kreativ um, ohne den Sinn zu verändern.
-   **Starke Sprache:** Verwende präzise Verben und Adjektive. Bevorzuge aktive Formulierungen gegenüber passiven.
-   **"Show, Don't Tell":** Zeige Emotionen, Charaktereigenschaften und Atmosphäre durch Handlungen, Dialoge und sensorische Details, anstatt sie nur zu behaupten.
-   **Dialog:** Schreibe glaubwürdige, lebendige Dialoge, die zur jeweiligen Figur passen und die Handlung vorantreiben oder Charaktertiefe verleihen.
-   **Setting & Atmosphäre:** Beschreibe den Ort, die Zeit und die Stimmung lebendig und immersiv. Nutze Details, die die Sinne ansprechen.
-   **Pacing & Spannung:** Baue die Handlung logisch auf, erzeuge Spannung und variiere das Tempo angemessen.
-   **Kohärenz & Konsistenz:** Erzähle eine in sich schlüssige Geschichte mit klarem Anfang, Mittelteil und Ende. Achte auf Konsistenz bei Zeitformen, Perspektive und Charakterdetails.
-   **Abgeschlossenes Ende:** Die Geschichte MUSS ein klares, befriedigendes und abgeschlossenes Ende haben. Sie darf NICHT abrupt oder mitten im Satz/Absatz enden.
-   **Formatierung:** Beginne NUR mit dem Titel im Format '# {titel}'. Füge KEINEN weiteren Titel hinzu. Strukturiere den Text sinnvoll mit Absätzen.

{zusatz_anweisungen}""",
            "PROMPT_OUTLINE_GEN": """Du bist ein erfahrener Autor und Story-Entwickler. Erstelle eine detaillierte Plot-Outline für eine Geschichte mit {kapitel_anzahl} Kapiteln basierend auf:
Titel: {titel}
Prompt: {prompt}
Setting: {setting}
Gesamtwortzahl: ca. {wortanzahl} Wörter ({sprache})

Erstelle eine Outline mit **klar voneinander abgegrenzten Abschnitten für jedes Kapitel**:
1.  Hauptcharaktere: Details (Name, Alter, Aussehen, Persönlichkeit, Motivation).
2.  Wichtige Nebencharaktere: Namen, Beziehung.
3.  Haupthandlung: Detaillierte Zusammenfassung des Verlaufs über alle Kapitel hinweg.
4.  Weltregeln (falls relevant).
5.  Kapitelstruktur ({kapitel_anzahl} Kapitel): **Für jedes Kapitel (beginne jeden Abschnitt deutlich mit z.B. 'Kapitel X:' oder '## Kapitel X')**: Aussagekräftiger Titel, Zeit/Ort, Anwesende Charaktere, Wichtigste Handlungspunkte/Szenen (5-7, spezifisch für dieses Kapitel), Charakterentwicklung/Wendungen in diesem Kapitel, Cliffhanger (optional).

**Wichtig:** Die Outline soll eine Grundlage für eine **qualitativ hochwertige Geschichte** bilden. Achte auf Konsistenz, Kausalität, logischen Aufbau, Charakterentwicklung und Potenzial für spannendes Erzählen gemäß hoher Schreibstandards (abwechslungsreiche Sprache, Vermeidung von Wiederholungen im Konzept etc.). Die Kapitelabschnitte müssen klar erkennbar sein.
{zusatz_anweisungen}""",

            # --- **FINAL REVISED** Chapter Prompt with Hierarchy ---
            "PROMPT_CHAPTER_GEN": """Du bist ein talentierter und erfahrener Autor von Kurzgeschichten in **deutscher Sprache**. Deine Aufgabe ist es, **Kapitel {kapitel_nummer} von {kapitel_anzahl}** einer Geschichte herausragend zu schreiben, indem du die bereitgestellten Kontexte **präzise und hierarchisch** nutzt.

**Kontext und Anweisungen für Kapitel {kapitel_nummer}:**

1.  **WAS SOLL PASSIEREN? (Primärer Fokus!)** - Halte dich **strikt** an die Handlungspunkte für **dieses spezifische Kapitel**, wie sie in folgendem Auszug aus der Plot-Outline beschrieben sind:
    --- START OUTLINE-AUSZUG KAPITEL {kapitel_nummer} ---
    {plot_outline_segment}
    --- ENDE OUTLINE-AUSZUG ---
    Setze **nur** diese Punkte um. Weiche nicht ab, füge keine Elemente aus anderen Kapiteln hinzu.

2.  **WAS IST BEREITS PASSIERT? (Globale Konsistenz & Anti-Looping!)** - Beachte die folgende Zusammenfassung der **gesamten bisherigen Handlung**. Stelle sicher, dass deine Erzählung dazu konsistent ist. **Wiederhole KEINE bereits abgeschlossenen Haupthandlungsstränge oder aufgelösten Kernkonflikte**, die in dieser Zusammenfassung erwähnt werden! Achte besonders auf den Status wichtiger Plotpunkte.
    --- START LAUFENDE GESAMTZUSAMMENFASSUNG ---
    {running_plot_summary}
    --- ENDE LAUFENDE GESAMTZUSAMMENFASSUNG ---

3.  **WIE ENDETE DAS LETZTE KAPITEL? (Direkter Anschluss!)** - Beginne **nahtlos** im Anschluss an das Ende des vorherigen Kapitels ({prev_kapitel_nummer}), wie im folgenden Detailkontext (Zusammenfassung + Raw End) beschrieben. Formuliere den Übergang neu und kopiere keine Sätze.
    --- START KONTEXT KAPITEL {prev_kapitel_nummer} ---
    {zusammenfassung_vorher}
    --- ENDE KONTEXT KAPITEL {prev_kapitel_nummer} ---

**Weitere Qualitätsrichtlinien:**
-   Umfang: ca. {kapitel_wortanzahl} Wörter (mind. {min_kapitel_worte}, max. {max_kapitel_worte}).
-   Sprache & Stil: Klar, prägnant, ansprechend. **Starke Variation** in Satzlänge, Struktur und Wortwahl. Exzellenter Lesefluss.
-   **Wiederholungen STRENG VERMEIDEN:** Vermeide nicht nur Phrasen, sondern auch **ähnliche Satzmuster, Beschreibungen und Handlungsmuster**. {zusatz_anweisungen}
-   Starke Sprache, "Show, Don't Tell", glaubwürdige Dialoge, Atmosphäre, Pacing (gemäß Outline-Segment).
-   Formatierung: Beginne mit '## Kapitel {kapitel_nummer}: [Passender Titel]' (AUSSER Kapitel 1: Beginne mit '# {titel}\\n\\n## Kapitel 1: [Kapiteltitel]'). Kein Titel am Ende.
-   Abschluss: Vollständiger, sinnvoller Satz/Absatz.

**Schreibe jetzt Kapitel {kapitel_nummer} gemäß ALLEN Anweisungen:**
""",

            # --- **FINAL REVISED** Epilogue Prompt ---
            "PROMPT_EPILOG_GEN": """Du bist ein talentierter Autor. Schreibe einen **hochwertigen, gut geschriebenen** und befriedigenden Abschluss (Epilog) für eine Geschichte.

**Kontext:**
-   **Titel:** {titel}
-   **Plot-Outline (Gesamtübersicht):**
    {plot_outline}
-   **Laufende Zusammenfassung der GESAMTEN Handlung bis zum Ende des letzten Kapitels:**
    {running_plot_summary}
-   **Detaillierter Kontext & Raw End vom LETZTEN Kapitel ({kapitel_anzahl}):**
    {zusammenfassung_vorher}
-   **Originaltext-Fragment vom Ende des letzten Kapitels:**
    ...{letztes_kapitel_ende}

**Deine Aufgabe für den Epilog:**
1.  **Auflösung:** Löse die wichtigsten offenen Handlungsstränge und Charakterbögen elegant auf, basierend auf der **Outline** und der **laufenden Zusammenfassung**.
2.  **Anknüpfung:** Knüpfe stilistisch und thematisch an den **detaillierten Kontext** des letzten Kapitels an.
3.  **Charakterende:** Führe die emotionale Reise der Hauptcharaktere zu einem sinnvollen Ende.
4.  **Themen:** Greife die Hauptthemen der Geschichte nochmals auf.
5.  **Qualität:** Schreibe stilistisch passend (abwechslungsreich, keine Wiederholungen). {zusatz_anweisungen}
6.  **Umfang:** ca. 500-1000 Wörter.
7.  **Formatierung:** Beginne mit '## Epilog'. Kein Titel am Ende.
8.  **Ende:** Sorge für ein klares, befriedigendes Ende.

**Schreibe jetzt den Epilog:**
""",

            # --- **FINAL REVISED** Detailed Summary Prompt ---
            "PROMPT_SUMMARY_GEN": """Du bist ein Assistent, der darauf spezialisiert ist, Kapitel einer Geschichte prägnant zusammenzufassen, um die Kontinuität für den Autor (eine KI) zu gewährleisten.

**Aufgabe:** Lies den folgenden Kapiteltext sorgfältig durch und erstelle eine kurze, prägnante Zusammenfassung (ca. 150-250 Wörter) in **deutscher Sprache**.

**Fokus der Zusammenfassung:**
-   Was sind die **wichtigsten Ereignisse**, die in diesem Kapitel stattgefunden haben?
-   Welche **signifikanten Entwicklungen** gab es bei den Hauptcharakteren (Entscheidungen, Erkenntnisse, emotionale Zustände)?
-   Wie ist der **genaue Zustand am Ende des Kapitels**? (Wo befinden sich die Charaktere? Was ist die unmittelbare Situation? Welche Spannung oder offene Frage besteht?)
-   **WICHTIG:** Was ist der **aktuelle Status wichtiger offener Fragen oder Kernkonflikte** (z.B. Wurde ein entscheidendes Ritual durchgeführt? Wurde ein Hauptgeheimnis gelüftet? Wurde ein Hauptziel erreicht?), basierend auf den Ereignissen *dieses* Kapitels? Benenne den Status klar (z.B. "Ritual abgeschlossen", "Geheimnis X ungelöst", "Ziel Y näher gekommen").
-   Vermeide Nebensächlichkeiten und konzentriere dich auf Informationen, die für das Verständnis des nächsten Kapitels **unbedingt notwendig** sind.

**Kapiteltext zum Zusammenfassen:**
--- START TEXT ---
{kapitel_text}
--- END TEXT ---

**Deine Zusammenfassung (nur der zusammenfassende Text):**
""",

            # --- **FINAL REVISED** Running Summary Update Prompt ---
            "PROMPT_RUNNING_SUMMARY_UPDATE": """Du bist ein Redakteur, der eine KI beim Schreiben einer Geschichte unterstützt. Deine Aufgabe ist es, eine **laufende Zusammenfassung der gesamten bisherigen Handlung** zu pflegen und zu aktualisieren.

**Bisherige Handlungszusammenfassung:**
{bisherige_zusammenfassung}

**Text des NEUEN Kapitels ({kapitel_nummer}), das gerade hinzugefügt wurde:**
--- START NEUES KAPITEL ---
{neues_kapitel_text}
--- END NEUES KAPITEL ---

**Deine Aufgabe:**
Aktualisiere die "Bisherige Handlungszusammenfassung" prägnant, indem du die **wichtigsten neuen Ereignisse, Charakterentwicklungen und Plotpunkte** aus dem NEUEN Kapitel hinzufügst. Behalte den roten Faden bei und **stelle klar den Fortschritt oder Abschluss wichtiger Handlungsbögen dar**. Die neue Zusammenfassung sollte die gesamte Geschichte bis einschließlich des neuen Kapitels abdecken und den **aktuellen Status der Hauptkonflikte/Ziele** widerspiegeln. Halte sie so kurz wie möglich, aber stelle sicher, dass alle *wesentlichen* Plotpunkte enthalten sind.

**Aktualisierte Gesamtzusammenfassung (nur der Text der neuen Zusammenfassung):**
""",

            # --- Other texts (ensure all needed keys are present) ---
            "USER_PROMPT_STORY": "Bitte schreibe eine hochwertige Kurzgeschichte basierend auf Titel '{titel}', Prompt und Setting. Ca. {wortanzahl} Wörter.",
            "USER_PROMPT_OUTLINE": "Bitte erstelle eine detaillierte Plot-Outline für eine {kapitel_anzahl}-teilige, qualitativ hochwertige Geschichte mit Titel '{titel}'.",
            "USER_PROMPT_CHAPTER": "Bitte schreibe Kapitel {kapitel_nummer} der Geschichte '{titel}' herausragend und gemäß allen Qualitätsrichtlinien, insbesondere unter Beachtung der hierarchischen Kontext-Anweisungen.",
            "USER_PROMPT_EPILOG": "Bitte schreibe einen hochwertigen Epilog für die Geschichte '{titel}' unter Berücksichtigung des Kontextes des letzten Kapitels und der Gesamthandlung.",
            "USER_PROMPT_SUMMARY": "Bitte fasse das vorherige Kapitel gemäß den Anweisungen zusammen, um die Kontinuität zu sichern.",
            "USER_PROMPT_RUNNING_SUMMARY": "Bitte aktualisiere die laufende Zusammenfassung der Geschichte mit den Ereignissen aus dem neuesten Kapitel.",
            "DEFAULT_CHAPTER_TITLE": "Kapitel {kapitel_nummer}", # Base name, formatted later
            "FIRST_CHAPTER_TITLE": "Kapitel {kapitel_nummer}", # Base name, formatted to 1 later
            "EPILOG_TITLE": "Epilog", # Base name
            "ERROR_CHAPTER_TITLE": "Kapitel {kapitel_nummer}: [Fehler bei der Generierung]", # Internal use
            "ERROR_CHAPTER_CONTENT": "Es ist ein Fehler bei der Generierung dieses Kapitels aufgetreten.", # Internal use
            "RESCUED_CHAPTER_NOTICE": "[Das Kapitel wurde aufgrund technischer Probleme gekürzt oder konnte nicht vollständig wiederhergestellt werden. Die Geschichte wird im nächsten Kapitel fortgesetzt.]",
            "RESCUED_EPILOG_NOTICE": "[Die Geschichte konnte nicht vollständig abgeschlossen werden. Der Epilog fehlt oder ist unvollständig.]",
            "SUMMARY_FIRST_CHAPTER": "Dies ist das erste Kapitel. Es gibt keinen vorherigen Kontext.", # More explicit
            "WARN_LOW_WORDCOUNT": "Warnung: Ziel-Wortanzahl {wortanzahl} sehr niedrig. Setze auf Minimum {min_val}.",
            "WARN_HIGH_WORDCOUNT": "Warnung: Ziel-Wortanzahl {wortanzahl} zu hoch ohne Kapitel-Modus. Setze auf Maximum {max_val}.",
            "INFO_CHAPTER_MODE_ACTIVATED": "Aktiviere Kapitel-Modus für Geschichte mit ca. {wortanzahl} Wörtern.",
            "INFO_GENERATING_STORY": "Generiere Geschichte '{titel}' mit ca. {wortanzahl} Wörtern in {sprache} (Qualitätsfokus)...",
            "INFO_SENDING_API_REQUEST": "Sende Anfrage an API (Model: {model}, Max Tokens: {max_tokens}, Temp: {temp:.2f})...",
            "INFO_WAITING_API_RESPONSE": "Warte auf API-Antwort...",
            "INFO_GENERATION_COMPLETE": "Generierung abgeschlossen nach {dauer:.1f} Sekunden.",
            "INFO_SAVED_TEMP_FILE": "Zwischendaten in temporäre Datei gespeichert: {dateiname}", # Debugging
            "WARN_CANNOT_WRITE_TEMP": "Warnung: Konnte temporäre Datei nicht schreiben: {error}", # Debugging
            "WARN_CANNOT_DELETE_TEMP": "Warnung: Konnte temporäre Datei nicht löschen: {error}", # Debugging
            "INFO_REMOVED_INCOMPLETE_SENTENCE": "Unvollständiger Satz am Ende entfernt.",
            "INFO_CORRECTING_ENDING": "Textende wird korrigiert...",
            "INFO_REMOVED_INCOMPLETE_DIALOG": "Unvollständigen Dialog am Ende entfernt.",
            "INFO_ADDED_MISSING_QUOTE": "Fehlendes Anführungszeichen ergänzt.",
            "INFO_REMOVED_INCOMPLETE_PARAGRAPH": "Unvollständigen letzten Absatz entfernt.",
            "INFO_GENERATING_CHAPTERS": "Generiere Geschichte in {kapitel_anzahl} Kapiteln (Qualitätsfokus)...",
            "INFO_GENERATING_OUTLINE": "Generiere Plot-Outline...",
            "INFO_OUTLINE_CREATED": "Plot-Outline erstellt.",
            "INFO_SAVED_OUTLINE_TEMP": "Plot-Outline temporär gespeichert: {dateiname}", # Debugging Temp Save
            "INFO_SAVED_OUTLINE_FINAL": "Plot-Outline gespeichert als: {dateiname}", # Final Save
            "WARN_CANNOT_SAVE_OUTLINE_TEMP": "Warnung: Konnte temporäre Plot-Outline nicht speichern: {error}", # Debugging Temp Save
            "ERROR_SAVING_OUTLINE_FINAL": "Fehler beim Speichern der Plot-Outline: {error}. Generierung wird fortgesetzt.", # Final Save
            "INFO_GENERATING_CHAPTER_NUM": "Generiere Kapitel {kapitel_nummer} von {kapitel_anzahl}...",
            "INFO_CHAPTER_COMPLETE": "Kapitel {kapitel_nummer} abgeschlossen ({dauer:.1f}s).",
            "ERROR_API_REQUEST_CHAPTER": "API-Fehler bei Kapitel {kapitel_nummer}: {error}",
            "INFO_RESCUED_PARTIAL_CONTENT": "Teilweise generierter Inhalt ({chars} Zeichen) gerettet.",
            "ERROR_READING_TEMP_FILE": "Fehler beim Lesen der temporären Datei: {error}", # Debugging
            "INFO_LAST_CHAPTER_INCOMPLETE": "Letztes Kapitel wirkt unvollständig oder wurde gerettet. Generiere Epilog...",
            "INFO_GENERATING_EPILOG": "Generiere Epilog...",
            "INFO_EPILOG_GENERATED": "Epilog generiert ({dauer:.1f}s).",
            "ERROR_GENERATING_EPILOG": "Fehler bei Generierung des Epilogs: {error}",
            "INFO_FINAL_WORD_COUNT": "Finale Geschichte mit {wortanzahl} Wörtern generiert.",
            "INFO_SAVED_TEXT_FILE": "Geschichte als Textdatei gespeichert: {dateiname}", # Used by CLI save_as_text_file
            "ERROR_API_OVERLOAD": "API überlastet/Timeout. Warte {delay:.1f}s, Versuch {retries}/{max_retries}...",
            "ERROR_API_CALL_FAILED": "Fehler während API-Aufruf: {error}",
            "ERROR_ALL_RETRIES_FAILED": "Alle API-Wiederholungen fehlgeschlagen.",
            "ERROR_UNSUPPORTED_LANGUAGE": "Sprache '{sprache}' nicht unterstützt. Unterstützt: {supported}",
            "ERROR_MISSING_API_KEY": "API-Schlüssel erforderlich. NEBIUS_API_KEY setzen oder übergeben.",
            "ERROR_GENERATION_FAILED": "Generierung der Geschichte fehlgeschlagen.",
            "COMMON_NOUN_PREFIXES": ["Der", "Die", "Das", "Ein", "Eine"], # Needed for _extract_char_names (still used internally by summary heuristic)
            "ACTION_VERBS": ["ging", "kam", "sprach", "sah", "fand", "entdeckte", "öffnete", "schloss", "rannte", "floh", "kämpfte", "starb", "tötete", "küsste", "sagte", "antwortete", "erwiderte", "blickte", "dachte"], # Needed for _extract_important_sentences (still used internally by summary heuristic)
            "EMOTIONAL_WORDS": ["angst", "furcht", "freude", "glück", "trauer", "wut", "zorn", "liebe", "hass", "entsetzen", "überraschung", "schock", "verzweiflung", "erleichterung"], # Needed for _extract_important_sentences (still used internally by summary heuristic)
            "CONJUNCTIONS_AT_END": ['und', 'aber', 'oder', 'denn', 'weil', 'dass', 'ob'],
            # --- Context Markers (Adopted from Web Script) ---
            "SUMMARY_LLM_PREFIX": "**Kontext - Zusammenfassung des vorherigen Kapitels (KI-generiert):**",
            "SUMMARY_RAW_END_MARKER_START": "\n\n**--- EXAKTES ENDE DES VORHERIGEN KAPITELS (KONTEXT) ---**",
            "SUMMARY_RAW_END_MARKER_END": "**--- ENDE KONTEXT ---**",
            "SUMMARY_FALLBACK": "[Kontext des vorherigen Kapitels konnte nicht automatisch generiert werden. Bitte die Handlung aus dem vorherigen Kapitel beachten.]",
            "INFO_GENERATING_SUMMARY": "Generiere LLM-Zusammenfassung + End-Extrakt für Kapitel {kapitel_nummer}...",
            "INFO_SUMMARY_GENERATED": "LLM-Zusammenfassung + End-Extrakt für Kapitel {kapitel_nummer} erstellt.",
            "ERROR_GENERATING_SUMMARY": "Fehler bei Generierung der LLM-Zusammenfassung für Kapitel {kapitel_nummer}: {error}",
            # --- Running Summary (Adopted from Web Script) ---
            "RUNNING_SUMMARY_PLACEHOLDER": "**Laufende Zusammenfassung der gesamten bisherigen Handlung:**\n",
            "RUNNING_SUMMARY_INITIAL": "Die Geschichte beginnt.", # Initial text for empty summary
            "INFO_UPDATING_RUNNING_SUMMARY": "Aktualisiere laufende Handlungszusammenfassung nach Kapitel {kapitel_nummer}...",
            "INFO_RUNNING_SUMMARY_UPDATED": "Laufende Handlungszusammenfassung aktualisiert.",
            "ERROR_UPDATING_RUNNING_SUMMARY": "Fehler beim Aktualisieren der laufenden Zusammenfassung nach Kapitel {kapitel_nummer}: {error}",
            "ERROR_GENERATING_OUTLINE_FALLBACK": "Generierung der Plot-Outline fehlgeschlagen", # Fallback text for outline
        },
        "Englisch": {
             # --- Story/Outline Prompts ---
            "PROMPT_STORY_GEN": """You are a talented and experienced author of short stories in **English**. Your task is to create a compelling, well-written story based on the following specifications.

Target word count: approx. {wortanzahl} words.

Title: {titel}
Prompt (Basic Idea): {prompt}
Setting (Location, Time, Atmosphere): {setting}

**Quality Guidelines for Writing:**
-   **Style & Flow:** Write in clear, concise, and engaging English. Vary sentence length and structure to avoid monotony. Ensure a natural reading flow.
-   **AVOID Repetition:** Pay VERY CLOSE attention to minimizing word and phrase repetitions as well as similar sentence patterns. Use synonyms or creatively rephrase sentences without altering the meaning.
-   **Strong Language:** Use precise verbs and evocative adjectives. Prefer active voice over passive.
-   **"Show, Don't Tell":** Demonstrate emotions, character traits, and atmosphere through actions, dialogue, and sensory details, rather than just stating them.
-   **Dialogue:** Write believable, vivid dialogue that fits the character and either advances the plot or reveals personality.
-   **Setting & Atmosphere:** Describe the location, time, and mood vividly and immersively. Use details that appeal to the senses.
-   **Pacing & Suspense:** Structure the plot logically, build suspense, and vary the pace appropriately.
-   **Coherence & Consistency:** Tell a cohesive story with a clear beginning, middle, and end. Ensure consistency in tense, perspective, and character details.
-   **Complete Ending:** The story MUST have a clear, satisfying, and conclusive ending. It must NOT end abruptly or mid-sentence/paragraph.
-   **Formatting:** Start ONLY with the title in the format '# {titel}'. Do NOT add another title. Structure the text meaningfully with paragraphs.

{zusatz_anweisungen}""",
            "PROMPT_OUTLINE_GEN": """You are a seasoned author and story developer. Create a detailed plot outline for a story with {kapitel_anzahl} chapters based on:
Title: {titel}
Prompt: {prompt}
Setting: {setting}
Total word count: approx. {wortanzahl} words ({sprache})

Create an outline with **clearly delineated sections for each chapter**:
1.  Main Characters: Details (name, age, appearance, personality, motivation).
2.  Important Side Characters: Names, relationships.
3.  Main Plot: Detailed summary of the storyline across all chapters.
4.  World Rules (if applicable).
5.  Chapter Structure ({kapitel_anzahl} chapters): **For each chapter (clearly start each section with e.g., 'Chapter X:' or '## Chapter X')**: Meaningful Title, Time/Location, Characters Present, Key Plot Points/Scenes (5-7, specific to this chapter), Character Development/Twists in this chapter, Cliffhanger (optional).

**Important:** The outline should serve as a foundation for a **high-quality story**. Ensure consistency, causality, logical progression, character development, and potential for engaging narrative adhering to high writing standards (varied language, avoiding repetition in the concept, etc.). Chapter sections must be clearly identifiable.
{zusatz_anweisungen}""",

            # --- **FINAL REVISED** Chapter Prompt with Hierarchy ---
            "PROMPT_CHAPTER_GEN": """You are a talented and experienced author of short stories in **English**. Your task is to write **Chapter {kapitel_nummer} of {kapitel_anzahl}** of a story exceptionally well, using the provided contexts **precisely and hierarchically**.

**Context and Instructions for Chapter {kapitel_nummer}:**

1.  **WHAT SHOULD HAPPEN? (Primary Focus!)** - Adhere **strictly** to the plot points for **this specific chapter** as described in the following excerpt from the plot outline:
    --- START OUTLINE EXCERPT CHAPTER {kapitel_nummer} ---
    {plot_outline_segment}
    --- END OUTLINE EXCERPT ---
    Implement **only** these points. Do not deviate, add elements from other chapters, or foreshadow excessively unless the outline implies it.

2.  **WHAT HAS ALREADY HAPPENED? (Global Consistency & Anti-Looping!)** - Refer to the following summary of the **entire plot so far**. Ensure your narrative is consistent with it. **Do NOT repeat major plot arcs or resolved core conflicts** mentioned in this summary! Pay close attention to the status of key plot points.
    --- START RUNNING OVERALL SUMMARY ---
    {running_plot_summary}
    --- END RUNNING OVERALL SUMMARY ---

3.  **HOW DID THE LAST CHAPTER END? (Direct Connection!)** - Begin **seamlessly** following the end of the previous chapter ({prev_kapitel_nummer}) as described in the detailed context below (summary + raw end). Rephrase the transition and do not copy sentences.
    --- START CONTEXT CHAPTER {prev_kapitel_nummer} ---
    {zusammenfassung_vorher}
    --- END CONTEXT CHAPTER {prev_kapitel_nummer} ---

**Further Quality Guidelines:**
-   Length: Approx. {kapitel_wortanzahl} words (min {min_kapitel_worte}, max {max_kapitel_worte}).
-   Language & Style: Clear, concise, engaging. **Strong variation** in sentence length, structure, and vocabulary. Excellent reading flow.
-   **STRICTLY AVOID Repetition:** Avoid repeating not just phrases, but also **similar sentence patterns, descriptions, and plot patterns**. {zusatz_anweisungen}
-   Strong language, "Show, Don't Tell", believable dialogue, atmosphere, pacing (according to outline segment).
-   Formatierung: Beginne mit '## Chapter {kapitel_nummer}: [Appropriate Title]' (Exception: '# {titel}\\n\\n## Chapter 1: [Chapter Title]'). No title at the end.
-   Conclusion: Complete, meaningful sentence/paragraph.

**Write Chapter {kapitel_nummer} now, following ALL instructions:**
""",

            # --- **FINAL REVISED** Epilogue Prompt ---
            "PROMPT_EPILOG_GEN": """You are a talented author. Write a **high-quality, well-written,** and satisfying conclusion (Epilogue) for a story.

**Context:**
-   **Title:** {titel}
-   **Plot Outline (Overall overview):**
    {plot_outline}
-   **Running Summary of the ENTIRE Plot up to the end of the last chapter:**
    {running_plot_summary}
-   **Detailed Context & Raw End from the LAST Chapter ({kapitel_anzahl}):**
    {zusammenfassung_vorher}
-   **Original Text Fragment from the end of the last chapter:**
    ...{letztes_kapitel_ende}

**Your Task for the Epilogue:**
1.  **Resolution:** Elegantly resolve the most important open plot threads and character arcs, based on the **Outline** and the **running summary**.
2.  **Connection:** Connect stylistically and thematically to the **detailed context** of the last chapter.
3.  **Character Ending:** Bring the main characters' emotional journey to a meaningful end.
4.  **Themes:** Revisit the main themes of the story.
5.  **Quality:** Write in a matching style (varied language, no repetition). {zusatz_anweisungen}
6.  **Length:** Approx. 500-1000 words.
7.  **Formatting:** Start with '## Epilogue'. No title at the end.
8.  **Ending:** Ensure a clear, satisfying ending.

**Write the Epilogue now:**
""",

            # --- **FINAL REVISED** Detailed Summary Prompt ---
            "PROMPT_SUMMARY_GEN": """You are an assistant specialized in concisely summarizing story chapters to ensure continuity for the author (an AI).

**Task:** Carefully read the following chapter text and create a brief, concise summary (approx. 150-250 words) in **English**.

**Focus of the Summary:**
-   What are the **most important events** that occurred in this chapter?
-   What **significant developments** happened with the main characters (decisions, realizations, emotional states)?
-   What is the **exact state at the end of the chapter**? (Where are the characters? What is the immediate situation? What tension or open question remains?)
-   **IMPORTANT:** What is the **current status of key open questions or core conflicts** (e.g., Was a crucial ritual performed? Was a main secret revealed? Was a primary goal achieved?) based on the events *in this* chapter? State the status clearly (e.g., "Ritual completed", "Mystery X unsolved", "Goal Y approached").
-   Avoid minor details and focus on information that is **absolutely essential** for understanding the next chapter.

**Chapter Text to Summarize:**
--- START TEXT ---
{kapitel_text}
--- END TEXT ---

**Your Summary (only the summary text):**
""",

            # --- **FINAL REVISED** Running Summary Update Prompt ---
            "PROMPT_RUNNING_SUMMARY_UPDATE": """You are an editor assisting an AI in writing a story. Your task is to maintain and update a **running summary of the entire plot so far**.

**Previous Plot Summary:**
{bisherige_zusammenfassung}

**Text of the NEW Chapter ({kapitel_nummer}) that was just added:**
--- START NEW CHAPTER ---
{neues_kapitel_text}
--- END NEW CHAPTER ---

**Your Task:**
Concisely update the "Previous Plot Summary" by adding the **most important new events, character developments, and plot points** from the NEW chapter. Maintain the narrative thread and **clearly state the progress or completion of major plot arcs**. The new summary should cover the entire story up to and including the new chapter and reflect the **current status of the main conflicts/goals**. Keep it as brief as possible, but ensure all *essential* plot points are included.

**Updated Overall Summary (only the text of the new summary):**
""",

            # --- Other texts (ensure all needed keys are present) ---
            "USER_PROMPT_STORY": "Please write a high-quality short story based on title '{titel}', prompt, and setting. Approx. {wortanzahl} words.",
            "USER_PROMPT_OUTLINE": "Please create a detailed plot outline for a {kapitel_anzahl}-chapter, high-quality story titled '{titel}'.",
            "USER_PROMPT_CHAPTER": "Please write chapter {kapitel_nummer} of the story '{titel}' exceptionally well, following all quality guidelines, especially adhering to the hierarchical context instructions.",
            "USER_PROMPT_EPILOG": "Please write a high-quality epilogue for the story '{titel}', considering the context from the last chapter and the overall plot.",
            "USER_PROMPT_SUMMARY": "Please summarize the previous chapter according to the instructions to ensure continuity.",
            "USER_PROMPT_RUNNING_SUMMARY": "Please update the running story summary with the events from the latest chapter.",
            "DEFAULT_CHAPTER_TITLE": "Chapter {kapitel_nummer}", # Base name
            "FIRST_CHAPTER_TITLE": "Chapter {kapitel_nummer}", # Base name, formatted to 1 later
            "EPILOG_TITLE": "Epilogue", # Base name
            "ERROR_CHAPTER_TITLE": "Chapter {kapitel_nummer}: [Generation Error]", # Internal use
            "ERROR_CHAPTER_CONTENT": "An error occurred during the generation of this chapter.", # Internal use
            "RESCUED_CHAPTER_NOTICE": "[This chapter was shortened due to technical issues or could not be fully recovered. The story continues in the next chapter.]",
            "RESCUED_EPILOG_NOTICE": "[The story could not be fully completed. The epilogue is missing or incomplete.]",
            "SUMMARY_FIRST_CHAPTER": "This is the first chapter. No previous context exists.", # More explicit
            "WARN_LOW_WORDCOUNT": "Warning: Target word count {wortanzahl} very low. Setting to minimum {min_val}.",
            "WARN_HIGH_WORDCOUNT": "Warning: Target word count {wortanzahl} too high without chapter mode. Setting to maximum {max_val}.",
            "INFO_CHAPTER_MODE_ACTIVATED": "Activating chapter mode for story with approx. {wortanzahl} words.",
            "INFO_GENERATING_STORY": "Generating story '{titel}' with approx. {wortanzahl} words in {sprache} (Quality Focus)...",
            "INFO_SENDING_API_REQUEST": "Sending request to API (Model: {model}, Max Tokens: {max_tokens}, Temp: {temp:.2f})...",
            "INFO_WAITING_API_RESPONSE": "Waiting for API response...",
            "INFO_GENERATION_COMPLETE": "Generation completed in {dauer:.1f} seconds.",
            "INFO_SAVED_TEMP_FILE": "Intermediate data saved to temporary file: {dateiname}", # Debugging
            "WARN_CANNOT_WRITE_TEMP": "Warning: Could not write temporary file: {error}", # Debugging
            "WARN_CANNOT_DELETE_TEMP": "Warning: Could not delete temporary file: {error}", # Debugging
            "INFO_REMOVED_INCOMPLETE_SENTENCE": "Removed incomplete sentence at the end.",
            "INFO_CORRECTING_ENDING": "Correcting text ending...",
            "INFO_REMOVED_INCOMPLETE_DIALOG": "Removed incomplete dialogue at the end.",
            "INFO_ADDED_MISSING_QUOTE": "Added missing quotation mark.",
            "INFO_REMOVED_INCOMPLETE_PARAGRAPH": "Removed incomplete last paragraph.",
            "INFO_GENERATING_CHAPTERS": "Generating story in {kapitel_anzahl} chapters (Quality Focus)...",
            "INFO_GENERATING_OUTLINE": "Generating plot outline...",
            "INFO_OUTLINE_CREATED": "Plot outline created.",
            "INFO_SAVED_OUTLINE_TEMP": "Plot outline temporarily saved: {dateiname}", # Debugging Temp Save
            "INFO_SAVED_OUTLINE_FINAL": "Plot outline saved as: {dateiname}", # Final Save
            "WARN_CANNOT_SAVE_OUTLINE_TEMP": "Warning: Could not save temporary plot outline: {error}", # Debugging Temp Save
            "ERROR_SAVING_OUTLINE_FINAL": "Error saving plot outline: {error}. Continuing generation.", # Final Save
            "INFO_GENERATING_CHAPTER_NUM": "Generating chapter {kapitel_nummer} of {kapitel_anzahl}...",
            "INFO_CHAPTER_COMPLETE": "Chapter {kapitel_nummer} completed ({dauer:.1f}s).",
            "ERROR_API_REQUEST_CHAPTER": "API error during chapter {kapitel_nummer}: {error}",
            "INFO_RESCUED_PARTIAL_CONTENT": "Rescued partially generated content ({chars} characters).",
            "ERROR_READING_TEMP_FILE": "Error reading temporary file: {error}", # Debugging
            "INFO_LAST_CHAPTER_INCOMPLETE": "Last chapter seems incomplete or was rescued. Generating epilogue...",
            "INFO_GENERATING_EPILOG": "Generating epilogue...",
            "INFO_EPILOG_GENERATED": "Epilogue generated ({dauer:.1f}s).",
            "ERROR_GENERATING_EPILOG": "Error generating epilogue: {error}",
            "INFO_FINAL_WORD_COUNT": "Final story generated with {wortanzahl} words.",
            "INFO_SAVED_TEXT_FILE": "Story saved as text file: {dateiname}", # Used by CLI save_as_text_file
            "ERROR_API_OVERLOAD": "API overloaded/timeout. Wait {delay:.1f}s, retry {retries}/{max_retries}...",
            "ERROR_API_CALL_FAILED": "Error during API call: {error}",
            "ERROR_ALL_RETRIES_FAILED": "All API retries failed.",
            "ERROR_UNSUPPORTED_LANGUAGE": "Language '{sprache}' not supported. Supported: {supported}",
            "ERROR_MISSING_API_KEY": "API key required. Set NEBIUS_API_KEY or pass directly.",
            "ERROR_GENERATION_FAILED": "Story generation failed.",
            "COMMON_NOUN_PREFIXES": ["The", "A", "An"], # Needed for _extract_char_names (if used internally)
            "ACTION_VERBS": ["went", "came", "spoke", "said", "answered", "replied", "saw", "found", "discovered", "opened", "closed", "ran", "fled", "fought", "died", "killed", "kissed", "looked", "thought", "realized"], # Needed for _extract_important_sentences (if used internally)
            "EMOTIONAL_WORDS": ["fear", "joy", "happiness", "sadness", "anger", "wrath", "love", "hate", "horror", "surprise", "shock", "despair", "relief"], # Needed for _extract_important_sentences (if used internally)
            "CONJUNCTIONS_AT_END": ['and', 'but', 'or', 'so', 'yet', 'because', 'if', 'that', 'when', 'while', 'although'],
             # --- Context Markers (Adopted from Web Script) ---
            "SUMMARY_LLM_PREFIX": "**Context - Summary of the previous chapter (AI-generated):**",
            "SUMMARY_RAW_END_MARKER_START": "\n\n**--- EXACT ENDING OF PREVIOUS CHAPTER (CONTEXT) ---**",
            "SUMMARY_RAW_END_MARKER_END": "**--- END CONTEXT ---**",
            "SUMMARY_FALLBACK": "[Context from the previous chapter could not be automatically generated. Please recall the events from the previous chapter.]",
            "INFO_GENERATING_SUMMARY": "Generating LLM summary + end extract for chapter {kapitel_nummer}...",
            "INFO_SUMMARY_GENERATED": "LLM summary + end extract for chapter {kapitel_nummer} created.",
            "ERROR_GENERATING_SUMMARY": "Error generating LLM summary for chapter {kapitel_nummer}: {error}",
            # --- Running Summary (Adopted from Web Script) ---
            "RUNNING_SUMMARY_PLACEHOLDER": "**Running Summary of the Entire Plot So Far:**\n",
            "RUNNING_SUMMARY_INITIAL": "The story begins.", # Initial text for empty summary
            "INFO_UPDATING_RUNNING_SUMMARY": "Updating running plot summary after chapter {kapitel_nummer}...",
            "INFO_RUNNING_SUMMARY_UPDATED": "Running plot summary updated.",
            "ERROR_UPDATING_RUNNING_SUMMARY": "Error updating running summary after chapter {kapitel_nummer}: {error}",
            "ERROR_GENERATING_OUTLINE_FALLBACK": "Plot outline generation failed", # Fallback text for outline
        }
    }

    def __init__(self, api_key: Optional[str] = None, model: str = MODELL_NAME, output_dir: str = "."):
        """Initializes the StoryGenerator."""
        self.resolved_api_key = api_key or os.environ.get("NEBIUS_API_KEY")
        lang_conf_de = self.LANGUAGE_CONFIG["Deutsch"] # Use German for init errors
        if not self.resolved_api_key:
            raise ValueError(lang_conf_de["ERROR_MISSING_API_KEY"])

        try:
            self.client = OpenAI(
                base_url=API_BASE_URL,
                api_key=self.resolved_api_key,
                timeout=300.0 # Longer timeout for potentially long story generation
            )
        except Exception as e:
            log.error(f"Error initializing OpenAI client: {e}")
            raise
        self.model_name = model
        self.output_dir = output_dir # Store output directory for saving outline
        self.total_steps = 0 # For progress bar
        self.current_step = 0 # For progress bar


    def _get_lang_config(self, sprache: str) -> Dict[str, Any]:
        """Gets the configuration for the specified language ('Deutsch' or 'Englisch')."""
        lang_lower = sprache.lower()
        if lang_lower in ['deutsch', 'german', 'de']:
            target_lang = "Deutsch"
        elif lang_lower in ['englisch', 'english', 'en']:
            target_lang = "Englisch"
        else:
            target_lang = sprache if sprache in self.LANGUAGE_CONFIG else None

        config = self.LANGUAGE_CONFIG.get(target_lang) if target_lang else None
        if not config:
             supported = ", ".join(SUPPORTED_LANGUAGES) # Use module-level constant
             lang_conf_de = self.LANGUAGE_CONFIG["Deutsch"] # Use German for the error itself
             raise ValueError(lang_conf_de["ERROR_UNSUPPORTED_LANGUAGE"].format(sprache=sprache, supported=supported))
        return config

    def _safe_filename(self, title: str) -> str:
        """Creates a safe filename from a title."""
        safe_title = re.sub(r'[^\w\s-]', '', title).strip()
        safe_title = re.sub(r'\s+', '_', safe_title)
        return safe_title[:50]

    def show_progress_bar(self):
        """Displays or updates the console progress bar."""
        if self.total_steps <= 0: return
        percent = float(self.current_step) / self.total_steps
        bar_length = 40
        arrow_length = max(0, int(round(percent * bar_length) - 1))
        arrow = '=' * arrow_length + ('>' if self.current_step < self.total_steps else '=')
        spaces = ' ' * (bar_length - len(arrow))
        percent_display = min(100, int(round(percent * 100)))
        sys.stdout.write(f"\rProgress: [{arrow}{spaces}] {percent_display}% ({self.current_step}/{self.total_steps})")
        sys.stdout.flush()
        if self.current_step >= self.total_steps:
            sys.stdout.write('\n')

    def _update_progress(self, step_increment=1):
        """Increments progress step and updates the bar."""
        self.current_step = min(self.current_step + step_increment, self.total_steps)
        self.show_progress_bar()

    def retry_api_call(self, call_function, *args, **kwargs):
        """Executes an API call with automatic retries on overload errors."""
        retries = 0
        lang_conf = self._get_lang_config("Deutsch") # Use German for generic messages
        while retries <= MAX_RETRIES:
            try:
                return call_function(*args, **kwargs)
            except Exception as e:
                # Convert specific OpenAI errors for better matching
                error_str = str(e).lower()
                error_type = type(e).__name__

                is_retryable = (
                    "overloaded" in error_str or
                    "rate limit" in error_str or
                    "timeout" in error_str or
                    "503" in error_str or
                    "504" in error_str or
                    "connection error" in error_str or
                    "service unavailable" in error_str or
                    isinstance(e, (OpenAI.APITimeoutError, OpenAI.APIConnectionError, OpenAI.RateLimitError, OpenAI.InternalServerError)) # Corrected attribute access
                )

                if is_retryable and retries < MAX_RETRIES:
                    retries += 1
                    current_retry_delay = (DEFAULT_RETRY_DELAY_S *
                                           (RETRY_BACKOFF_FACTOR ** (retries - 1)) +
                                           random.uniform(0.1, 1.0)) # Jitter
                    log.warning(lang_conf["ERROR_API_OVERLOAD"].format(
                        delay=current_retry_delay, retries=retries, max_retries=MAX_RETRIES
                    ) + f" (Type: {error_type})", exc_info=False) # Log error type
                    time.sleep(current_retry_delay)
                else:
                    # Log the final error with traceback if not retryable or retries exceeded
                    log.error(lang_conf["ERROR_API_CALL_FAILED"].format(error=str(e)) + f" (Type: {error_type})", exc_info=True)
                    raise # Re-raise the original exception
        # This part should ideally not be reached if the loop always raises on failure
        raise Exception(lang_conf["ERROR_ALL_RETRIES_FAILED"])


    def _create_system_prompt(self, sprache: str, template_key: str, context: Dict[str, Any]) -> str:
        """Creates a system prompt based on language and template."""
        lang_conf = self._get_lang_config(sprache)
        template = lang_conf.get(template_key)
        if not template:
            raise ValueError(f"Prompt template '{template_key}' not found for language '{sprache}'.")

        context['sprache'] = sprache # Ensure language is in context
        context.setdefault('zusatz_anweisungen', '')
        # Format additional instructions if provided
        if context.get('zusatz_anweisungen'): # Check if value is truthy
             if not context['zusatz_anweisungen'].strip().startswith("**Zusätzliche Anweisungen:**") and \
                not context['zusatz_anweisungen'].strip().startswith("**Additional Instructions:**"):
                 prefix = "**Zusätzliche Anweisungen:**" if sprache == "Deutsch" else "**Additional Instructions:**"
                 context['zusatz_anweisungen'] = f"\n{prefix}\n{context['zusatz_anweisungen'].strip()}\n"
             else:
                 context['zusatz_anweisungen'] = f"\n{context['zusatz_anweisungen'].strip()}\n"
        else:
             context['zusatz_anweisungen'] = ""

        # Ensure all required keys for the specific template are present or provide defaults
        required_keys = {
            "PROMPT_STORY_GEN": ['wortanzahl', 'titel', 'prompt', 'setting'],
            "PROMPT_OUTLINE_GEN": ['kapitel_anzahl', 'titel', 'prompt', 'setting', 'wortanzahl'],
            "PROMPT_CHAPTER_GEN": ['kapitel_nummer', 'kapitel_anzahl', 'titel', # Removed 'prompt', 'setting' as they are less critical here now
                                   'plot_outline_segment', # Uses segment
                                   'zusammenfassung_vorher', 'running_plot_summary',
                                   'kapitel_wortanzahl', 'min_kapitel_worte', 'max_kapitel_worte', 'prev_kapitel_nummer'],
            "PROMPT_EPILOG_GEN": ['titel', 'plot_outline', 'zusammenfassung_vorher', # Epilog uses full outline
                                  'running_plot_summary', 'letztes_kapitel_ende', 'kapitel_anzahl'],
            "PROMPT_SUMMARY_GEN": ['kapitel_text'],
            "PROMPT_RUNNING_SUMMARY_UPDATE": ['bisherige_zusammenfassung', 'neues_kapitel_text', 'kapitel_nummer']
        }

        defaults = {
            'wortanzahl': DEFAULT_TARGET_WORDS,
            'titel': 'Unbenannt' if sprache == "Deutsch" else 'Untitled',
            'prompt': '', 'setting': '', 'kapitel_anzahl': 0, 'kapitel_nummer': 0,
            'plot_outline': '[Keine Outline]' if sprache == "Deutsch" else '[No Outline]', # Default for Epilog
            'plot_outline_segment': '[Outline Segment nicht verfügbar]' if sprache == "Deutsch" else '[Outline Segment Unavailable]',
            'zusammenfassung_vorher': '', 'kapitel_wortanzahl': 1000,
            'min_kapitel_worte': 800, 'max_kapitel_worte': 1500,
            'letztes_kapitel_ende': '', 'kapitel_text': '',
            'running_plot_summary': lang_conf.get("RUNNING_SUMMARY_PLACEHOLDER", "") + lang_conf.get("RUNNING_SUMMARY_INITIAL", ""), # Default for new key
            'bisherige_zusammenfassung': lang_conf.get("RUNNING_SUMMARY_INITIAL", ""), 'neues_kapitel_text': '', # Defaults for new prompt
            'prev_kapitel_nummer': 0
        }

        # Set defaults for keys required by the current template
        if template_key in required_keys:
            for key in required_keys[template_key]:
                 context.setdefault(key, defaults.get(key))

        # Use .format_map for safer formatting
        try:
             # Add model name to context for potential use in prompts (optional)
             context['model_name'] = self.model_name
             return template.format_map(context)
        except KeyError as e:
             log.error(f"Missing key in prompt context for template '{template_key}': {e}. Context: {context}", exc_info=True)
             raise ValueError(f"Missing context for prompt template '{template_key}': {e}")


    def _generate_single_story(self, prompt: str, setting: str, titel: str,
                               word_count: int, sprache: str,
                               additional_instructions: Optional[str]
                               ) -> Optional[str]:
        """Generates a shorter story in a single API call with quality focus."""
        lang_conf = self._get_lang_config(sprache)
        temp_fd, temp_dateiname = tempfile.mkstemp(prefix="temp_story_", suffix=".txt")
        os.close(temp_fd)
        self._update_progress(0) # Initialize progress for single story

        try:
            system_prompt = self._create_system_prompt(
                sprache, "PROMPT_STORY_GEN", {
                    "wortanzahl": word_count, "titel": titel, "prompt": prompt,
                    "setting": setting, "zusatz_anweisungen": additional_instructions
                })

            max_tokens = min(int(word_count * TOKEN_WORD_RATIO * 1.1), MAX_TOKENS_PER_CALL)
            temperature = 0.75
            log.info(lang_conf["INFO_SENDING_API_REQUEST"].format(model=self.model_name, max_tokens=max_tokens, temp=temperature))
            start_time = time.time()
            log.info(lang_conf["INFO_WAITING_API_RESPONSE"] + " (Single Story)")
            user_prompt_text = lang_conf["USER_PROMPT_STORY"].format(titel=titel, wortanzahl=word_count)

            response = self.retry_api_call(
                self.client.chat.completions.create,
                model=self.model_name, max_tokens=max_tokens, temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt_text},
                ]
            )
            story = response.choices[0].message.content
            duration = time.time() - start_time
            log.info(lang_conf["INFO_GENERATION_COMPLETE"].format(dauer=duration))
            self._update_progress(1) # Complete progress for single story

            try:
                with open(temp_dateiname, 'w', encoding='utf-8') as f: f.write(story)
                log.debug(lang_conf["INFO_SAVED_TEMP_FILE"].format(dateiname=temp_dateiname))
            except Exception as e:
                log.warning(lang_conf["WARN_CANNOT_WRITE_TEMP"].format(error=str(e)))

            story = self._format_story(story, titel, sprache)
            story = self._clean_text_ending(story, sprache)
            return story

        except Exception as e:
            # Ensure progress bar finishes on error
            if self.current_step < self.total_steps: self._update_progress(self.total_steps - self.current_step)

            if not isinstance(e, (OpenAI.APIError, Exception)): # Avoid double logging if retry failed
                log.error(f"Unexpected error during single story generation for '{titel}': {str(e)}", exc_info=True)
            # Attempt rescue
            if os.path.exists(temp_dateiname):
                try:
                    with open(temp_dateiname, 'r', encoding='utf-8') as f:
                        partial_story = f.read()
                    if len(partial_story) > MIN_CHARS_FOR_RESCUE:
                        log.info(lang_conf["INFO_RESCUED_PARTIAL_CONTENT"].format(chars=len(partial_story)))
                        partial_story = self._format_story(partial_story, titel, sprache)
                        partial_story = self._clean_text_ending(partial_story, sprache)
                        notice_key = "RESCUED_EPILOG_NOTICE" # Use this key for general incompleteness
                        return partial_story + f"\n\n{lang_conf.get(notice_key, '[Story generation may be incomplete due to an error.]')}"
                except Exception as e2:
                    log.error(lang_conf["ERROR_READING_TEMP_FILE"].format(error=str(e2)))
            return None
        finally:
            if os.path.exists(temp_dateiname):
                try: os.remove(temp_dateiname)
                except Exception as e: log.warning(lang_conf["WARN_CANNOT_DELETE_TEMP"].format(error=str(e)))


    def _format_story(self, story: str, titel: str, sprache: str) -> str:
        """Ensures the story starts correctly with the title (H1)."""
        story = story.strip()
        titel_prefix = f"# {titel}" # Markdown H1 format
        if not story.startswith(titel_prefix):
            lines = story.split('\n')
            # Remove all lines at the beginning starting with #, except the expected title
            while lines and lines[0].strip().startswith("#") and lines[0].strip() != titel_prefix:
                log.debug(f"Removing incorrect leading line during format_story: {lines[0]}")
                lines.pop(0)
            story = "\n".join(lines).strip()
            # Add the correct title if it's still missing
            if not story.startswith(titel_prefix):
                 log.debug(f"Prepending main title '{titel_prefix}' during format_story")
                 story = f"{titel_prefix}\n\n{story}"
        return story

    def _clean_text_ending(self, text: str, sprache: str) -> str:
        """Checks and corrects abrupt endings in the text. (Adopted from Web version, generally more robust)"""
        if not text or len(text.strip()) < MIN_CHARS_FOR_RESCUE // 3:
            return text

        lang_conf = self._get_lang_config(sprache)
        original_text = text
        text = text.rstrip()

        # 1. Incomplete sentence ending (more conservative check)
        # Looks for patterns like "... word Word." at the very end
        match = re.search(r'[a-zäöüß][,;]?\s+[A-ZÄÖÜ][a-zäöüß]+\.?\s*$', text)
        if match:
            # Find last definite sentence end BEFORE the match
            last_sentence_end = -1
            potential_ends = ['. ', '! ', '? ', '." ', '!" ', '?" ', '\n\n']
            for p in potential_ends:
                found_pos = text.rfind(p, 0, match.start())
                if found_pos > last_sentence_end:
                     # Calculate actual end index after punctuation + space/quote
                     end_idx = found_pos + len(p)
                     if p.endswith('" ') and len(text) > end_idx and text[end_idx] == '"': end_idx +=1 # Handle ." " etc.
                     last_sentence_end = end_idx

            if last_sentence_end > 0:
                text = text[:last_sentence_end].strip() # Cut after the end marker
                log.info(lang_conf["INFO_REMOVED_INCOMPLETE_SENTENCE"])
            else:
                 log.debug("Could not reliably fix potentially incomplete sentence at the end.")


        # 2. No sentence-ending punctuation and ends with letter/number
        ends_with_punctuation = any(text.endswith(p) for p in ['.', '!', '?', '"', "'", '”', '’'])
        last_char_is_alphanum = text[-1].isalnum() if text else False

        if not ends_with_punctuation and last_char_is_alphanum:
             # Find the last punctuation mark anywhere before the end
             last_sentence_end = max(text.rfind('.'), text.rfind('!'), text.rfind('?'), text.rfind('\n\n'))
             if last_sentence_end > 0:
                 # Truncate after the last found punctuation/paragraph break
                 cutoff_point = last_sentence_end + 1
                 # Include trailing quotes/spaces after punctuation
                 while cutoff_point < len(text) and text[cutoff_point] in [' ', '"', "'", '”', '’']:
                       cutoff_point += 1
                 text = text[:cutoff_point].strip()
                 log.info(lang_conf["INFO_CORRECTING_ENDING"] + " (Missing punctuation/abrupt end)")
             else:
                 # Maybe it ends mid-sentence without prior punctuation? Try cutting last line.
                 last_newline = text.rfind('\n')
                 if last_newline > 0 and len(text) - last_newline < 50: # If last line is short
                      text = text[:last_newline].strip()
                      log.info(lang_conf["INFO_CORRECTING_ENDING"] + " (Removed potentially incomplete last line)")
                 else:
                      log.debug("Text ends without punctuation, and no prior punctuation found.")


        # 3. Incomplete dialogue quotes (Handles various quote types)
        paragraphs = text.split('\n\n')
        last_paragraph = paragraphs[-1].strip() if paragraphs else ""
        if last_paragraph:
            quote_chars = ['"', "'", '“', '”', '‘', '’']
            matching_quotes = {'"': '"', "'": "'", '“': '”', '‘': '’'}
            open_quote_stack = []

            for char in last_paragraph:
                if char in matching_quotes.keys(): # Opening quote
                    open_quote_stack.append(char)
                elif char in matching_quotes.values(): # Closing quote
                    # Check if it matches the last opened quote
                    if open_quote_stack and char == matching_quotes.get(open_quote_stack[-1]):
                        open_quote_stack.pop()
                    # else: Mismatched closing quote - ignore for ending check

            # If stack is not empty, a quote is unclosed
            if open_quote_stack:
                last_open_quote_char = open_quote_stack[-1]
                # Find the position of the last unclosed opening quote
                last_open_quote_index = last_paragraph.rfind(last_open_quote_char)

                # Find the last sentence end *within the paragraph* before the unclosed quote
                last_sentence_end_in_para = max(
                    last_paragraph.rfind('.', 0, last_open_quote_index),
                    last_paragraph.rfind('!', 0, last_open_quote_index),
                    last_paragraph.rfind('?', 0, last_open_quote_index)
                )

                # Check if the unclosed quote starts after the last sentence ends
                if last_open_quote_index > last_sentence_end_in_para:
                    # Cut off after the last complete sentence before the dangling dialogue
                    text_before_last_para = '\n\n'.join(paragraphs[:-1])
                    if text_before_last_para: text_before_last_para += '\n\n'

                    cutoff_point_in_para = last_sentence_end_in_para
                    if cutoff_point_in_para >= 0 : # Found punctuation before the dangling quote
                         # Include trailing space/quote after punctuation
                         cutoff_point_in_para += 1
                         while cutoff_point_in_para < last_open_quote_index and last_paragraph[cutoff_point_in_para] in [' ', '"', "'", '”', '’']:
                               cutoff_point_in_para += 1
                         text = text_before_last_para + last_paragraph[:cutoff_point_in_para].strip()
                         log.info(lang_conf["INFO_REMOVED_INCOMPLETE_DIALOG"])
                    else:
                        # No sentence end found before the quote in this paragraph, remove the whole para
                        text = text_before_last_para.strip()
                        log.info(lang_conf["INFO_REMOVED_INCOMPLETE_PARAGRAPH"] + " (Due to dangling quote start)")


        # 4. Short last paragraph or ending with conjunction
        paragraphs = text.split('\n\n') # Recalculate in case text changed
        if len(paragraphs) > 1:
            last_paragraph_words = paragraphs[-1].strip().split()
            # Get language-specific conjunctions
            conjunctions = lang_conf.get("CONJUNCTIONS_AT_END", [])
            ends_with_conj = False
            if last_paragraph_words:
                 last_word_cleaned = last_paragraph_words[-1].lower().strip('".!?,’”)')
                 ends_with_conj = last_word_cleaned in conjunctions

            # Check if the *second last* paragraph looks complete
            second_last_para = paragraphs[-2].strip()
            second_last_ends_ok = any(second_last_para.endswith(p) for p in ['.','!','?','"','\'','”','’'])

            # Remove last paragraph if it's short/ends with conjunction AND previous looks complete
            if (len(last_paragraph_words) < MIN_WORDS_FOR_VALID_ENDING or ends_with_conj) and second_last_ends_ok:
                text = '\n\n'.join(paragraphs[:-1]).strip()
                log.info(lang_conf["INFO_REMOVED_INCOMPLETE_PARAGRAPH"])
            elif len(last_paragraph_words) < 2 and not second_last_ends_ok:
                 # Also remove very short (0/1 word) last paragraphs if prev is also incomplete
                 text = '\n\n'.join(paragraphs[:-1]).strip()
                 log.info(lang_conf["INFO_REMOVED_INCOMPLETE_PARAGRAPH"] + " (Very short)")

        # Log if changes were made
        if text != original_text:
             log.info(f"Text ending cleaned. Length reduced from {len(original_text)} to {len(text)}.")
             log.debug(f"Cleaned End: ...{text[-80:]}")
        return text

    def _optimize_chapter_structure(self, word_count: int, max_words_per_chapter: int) -> Tuple[int, List[int]]:
        """Calculates a simple chapter structure. (Adopted from Web version)"""
        if word_count <= max_words_per_chapter * 1.5:
            # If only slightly above max, create 1 or 2 chapters
            num_chapters = max(1, math.ceil(word_count / max_words_per_chapter))
        else:
            # Otherwise, aim for a number based on total length or minimum required
            num_chapters = max(MIN_CHAPTERS_LONG_STORY,
                               round(word_count / TARGET_WORDS_PER_CHAPTER_DIVISOR))

        num_chapters = max(1, num_chapters) # Ensure at least 1

        base_words = word_count // num_chapters
        remainder = word_count % num_chapters
        words_per_chapter = [base_words] * num_chapters
        # Distribute remainder words to the first few chapters
        for i in range(remainder):
            words_per_chapter[i] += 1

        # Recalculate if any chapter significantly exceeds the max
        # Allow slightly more flexibility (e.g., 1.1x) before forcing recalculation
        max_allowed_flex = max_words_per_chapter * 1.1
        if any(w > max_allowed_flex for w in words_per_chapter):
            log.debug(f"Recalculating chapter structure as limit {max_words_per_chapter} (~{max_allowed_flex:.0f}) was exceeded.")
            num_chapters += 1 # Just add one more chapter
            base_words = word_count // num_chapters
            remainder = word_count % num_chapters
            words_per_chapter = [base_words] * num_chapters
            for i in range(remainder): words_per_chapter[i] += 1
            # Check again, log warning if still problematic (e.g., extremely high total count)
            if any(w > max_words_per_chapter * 1.2 for w in words_per_chapter): # Use 1.2x for warning
                 log.warning(f"Chapter structure could not strictly meet the max limit ({max_words_per_chapter}). Final distribution: {words_per_chapter}")

        # Ensure no chapter has 0 words (minimum practical size)
        words_per_chapter = [max(100, w) for w in words_per_chapter] # Set a minimum floor

        return num_chapters, words_per_chapter

    # Removed: _extract_character_names, _extract_important_sentences, _create_chapter_summary
    # Added: _extract_outline_segment, _generate_chapter_summary_llm, _update_running_summary_llm (Copied from Web version)

    def _extract_outline_segment(self, plot_outline: str, chapter_number: int, total_chapters: int, sprache: str) -> Optional[str]:
        """
        Attempts to extract the specific section for the given chapter_number
        from the full plot_outline text using regex. Returns the segment or None.
        Relies on headings like "Kapitel X:" or "## Kapitel X".
        """
        if not plot_outline: return None
        lang_conf = self._get_lang_config(sprache)
        if lang_conf.get("ERROR_GENERATING_OUTLINE_FALLBACK", "<ERR>") in plot_outline:
            return None

        chapter_word = "Kapitel" if sprache == "Deutsch" else "Chapter"

        # Regex: Optional leading whitespace/markdown, chapter word, number, optional colon/space/newline
        # Makes the chapter number mandatory \b ensures whole word match
        start_pattern = re.compile(
            rf"^[#\s]*{chapter_word}\s+{chapter_number}\b[:\s]*\n?",
            re.IGNORECASE | re.MULTILINE
        )

        next_chapter_num = chapter_number + 1
        # End pattern: Start of the next chapter OR common concluding words (Epilog, Fazit, etc.)
        # Added more potential end markers
        end_pattern = re.compile(
            rf"^[#\s]*(?:(?:{chapter_word}\s+{next_chapter_num}\b)|(?:Epilog|Fazit|Conclusion|Summary|Gesamtfazit|Final Thoughts))[:\s]*\n?",
            re.IGNORECASE | re.MULTILINE
        )

        start_match = start_pattern.search(plot_outline)
        if not start_match:
            # Fallback: Try finding just the number followed by a period or colon, e.g., "3." or "3:"
             start_pattern_num_only = re.compile(rf"^[#\s]*{chapter_number}[.:]\s*\n?", re.MULTILINE)
             start_match = start_pattern_num_only.search(plot_outline)
             if not start_match:
                log.warning(f"Could not find start pattern for Chapter {chapter_number} in outline.")
                return None # Return None if not found

        start_index = start_match.end() # Start *after* the found heading line

        # Find the start of the next relevant section or end of text
        end_match = end_pattern.search(plot_outline, pos=start_index)
        end_index = end_match.start() if end_match else len(plot_outline) # Go to end of string if no next section found

        segment = plot_outline[start_index:end_index].strip()

        if not segment:
             log.warning(f"Extracted empty outline segment for Chapter {chapter_number}. This might indicate an outline formatting issue.")
             # Try a simple paragraph grab as fallback? Risky. Return None is safer.
             return None

        log.debug(f"Extracted outline segment for Chapter {chapter_number} ({len(segment)} chars)")
        return segment


    def _generate_chapter_summary_llm(self, chapter_text: str, chapter_number: int, sprache: str) -> Tuple[str, str]:
        """
        Generates an LLM summary AND extracts the raw end of the chapter.
        Returns a tuple: (formatted_summary_with_prefix_and_markers, raw_end_text_only)
        """
        lang_conf = self._get_lang_config(sprache)
        log.info(lang_conf["INFO_GENERATING_SUMMARY"].format(kapitel_nummer=chapter_number))

        # --- 1. Extract Raw End ---
        raw_end_text_only = ""
        if chapter_text:
             chapter_text_stripped = chapter_text.strip()
             chars_for_raw_end = 800 # Target length for raw end context
             start_index = max(0, len(chapter_text_stripped) - chars_for_raw_end)
             raw_end_candidate = chapter_text_stripped[start_index:]

             # Try to start raw end from the beginning of the last paragraph if feasible
             last_para_break = raw_end_candidate.rfind('\n\n')
             # Use paragraph break if it's not too far back (e.g., > 30% into the candidate)
             if last_para_break > len(raw_end_candidate) * 0.3:
                 raw_end_text_only = raw_end_candidate[last_para_break:].strip()
             else:
                 raw_end_text_only = raw_end_candidate.strip()

             # Trim if excessively long (safety net)
             if len(raw_end_text_only) > chars_for_raw_end * 1.5:
                  raw_end_text_only = raw_end_text_only[-(int(chars_for_raw_end * 1.5)):]

             raw_end_text_only = raw_end_text_only.strip()
             log.debug(f"Extracted raw end for chapter {chapter_number} ({len(raw_end_text_only)} chars)")

        # --- 2. Generate LLM Summary ---
        # Default fallback in case generation fails
        llm_summary_part = f"{lang_conf['SUMMARY_LLM_PREFIX']}\n{lang_conf['SUMMARY_FALLBACK']}"

        if not chapter_text or len(chapter_text.strip()) < 100:
            log.warning(f"Chapter {chapter_number} text is too short or missing for LLM summary generation.")
            # Return fallback summary and whatever raw end we got
            combined_context = llm_summary_part
            if raw_end_text_only:
                 raw_end_marker_start = lang_conf.get("SUMMARY_RAW_END_MARKER_START", "")
                 raw_end_marker_end = lang_conf.get("SUMMARY_RAW_END_MARKER_END", "")
                 combined_context += f"{raw_end_marker_start}\n{raw_end_text_only}\n{raw_end_marker_end}"
            return (combined_context, raw_end_text_only)

        try:
            max_tokens_summary = 800 # Tokens just for the summary generation call
            temperature_summary = 0.55 # Lower temp for factual summary

            system_prompt = self._create_system_prompt(
                sprache, "PROMPT_SUMMARY_GEN", {"kapitel_text": chapter_text}
            )
            user_prompt = lang_conf["USER_PROMPT_SUMMARY"]

            log.info(lang_conf["INFO_SENDING_API_REQUEST"].format(model=self.model_name, max_tokens=max_tokens_summary, temp=temperature_summary) + " (Summary)")
            log.info(lang_conf["INFO_WAITING_API_RESPONSE"] + " (Summary)")

            response = self.retry_api_call(
                self.client.chat.completions.create,
                model=self.model_name,
                max_tokens=max_tokens_summary,
                temperature=temperature_summary,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            summary_text = response.choices[0].message.content.strip()
            # Clean up potential LLM boilerplate
            summary_text = re.sub(r'^(Zusammenfassung|Summary)[:\s]*', '', summary_text, flags=re.IGNORECASE | re.MULTILINE).strip()

            if not summary_text:
                 log.warning(f"LLM summary for chapter {chapter_number} was empty. Using fallback.")
                 # llm_summary_part remains the fallback defined earlier
            else:
                 log.info(lang_conf["INFO_SUMMARY_GENERATED"].format(kapitel_nummer=chapter_number))
                 llm_summary_part = f"{lang_conf['SUMMARY_LLM_PREFIX']}\n{summary_text}"

            # Combine LLM summary and raw end with markers for the final context string
            combined_context = llm_summary_part
            if raw_end_text_only:
                 raw_end_marker_start = lang_conf.get("SUMMARY_RAW_END_MARKER_START", "")
                 raw_end_marker_end = lang_conf.get("SUMMARY_RAW_END_MARKER_END", "")
                 combined_context += f"{raw_end_marker_start}\n{raw_end_text_only}\n{raw_end_marker_end}"

            # Return the combined string (for next chapter prompt) and the raw end separately (for epilogue)
            return (combined_context, raw_end_text_only)

        except Exception as e:
            log.error(lang_conf["ERROR_GENERATING_SUMMARY"].format(kapitel_nummer=chapter_number, error=str(e)), exc_info=True)
            # Return fallback summary and whatever raw end we got
            combined_context = llm_summary_part # This is already the fallback
            if raw_end_text_only:
                 raw_end_marker_start = lang_conf.get("SUMMARY_RAW_END_MARKER_START", "")
                 raw_end_marker_end = lang_conf.get("SUMMARY_RAW_END_MARKER_END", "")
                 combined_context += f"{raw_end_marker_start}\n{raw_end_text_only}\n{raw_end_marker_end}"
            return (combined_context, raw_end_text_only)


    def _update_running_summary_llm(self, previous_summary: str, new_chapter_text: str, chapter_number: int, sprache: str) -> str:
        """Updates the running plot summary using the latest chapter."""
        lang_conf = self._get_lang_config(sprache)
        log.info(lang_conf["INFO_UPDATING_RUNNING_SUMMARY"].format(kapitel_nummer=chapter_number))

        placeholder = lang_conf.get("RUNNING_SUMMARY_PLACEHOLDER", "<PLACEHOLDER>")
        initial_text = lang_conf.get("RUNNING_SUMMARY_INITIAL", "The story begins.")
        # Strip placeholder before sending to LLM
        if previous_summary.startswith(placeholder):
             current_summary_text = previous_summary[len(placeholder):].strip()
        else:
             current_summary_text = previous_summary.strip() # Should not happen, but safe

        # Handle the very first update
        if current_summary_text == initial_text:
            current_summary_text = "" # Start fresh for the first real summary

        if not new_chapter_text or len(new_chapter_text.strip()) < 50:
            log.warning(f"New chapter {chapter_number} text too short, skipping running summary update.")
            return previous_summary # Return the unmodified previous summary (with placeholder)

        try:
            # Adjust token limit based on estimated size of previous summary + chapter text?
            # For now, use a generous fixed limit.
            max_tokens_update = 1500 # Allow ample space for the updated summary
            temperature_update = 0.6 # Slightly higher temp for integration

            system_prompt = self._create_system_prompt(
                sprache, "PROMPT_RUNNING_SUMMARY_UPDATE", {
                    "bisherige_zusammenfassung": current_summary_text, # Pass text without placeholder
                    "neues_kapitel_text": new_chapter_text,
                    "kapitel_nummer": chapter_number
                }
            )
            user_prompt = lang_conf["USER_PROMPT_RUNNING_SUMMARY"]

            log.info(lang_conf["INFO_SENDING_API_REQUEST"].format(model=self.model_name, max_tokens=max_tokens_update, temp=temperature_update) + " (Running Summary Update)")
            log.info(lang_conf["INFO_WAITING_API_RESPONSE"] + " (Running Summary Update)")

            response = self.retry_api_call(
                self.client.chat.completions.create,
                model=self.model_name,
                max_tokens=max_tokens_update,
                temperature=temperature_update,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            updated_summary_text = response.choices[0].message.content.strip()
            # Clean up potential LLM boilerplate
            updated_summary_text = re.sub(r'^(Aktualisierte Gesamtzusammenfassung|Updated Overall Summary)[:\s]*', '', updated_summary_text, flags=re.IGNORECASE | re.MULTILINE).strip()

            if not updated_summary_text:
                 log.warning(f"LLM running summary update after chapter {chapter_number} was empty. Reverting to previous summary.")
                 return previous_summary # Return the original with placeholder
            else:
                 log.info(lang_conf["INFO_RUNNING_SUMMARY_UPDATED"])
                 # Basic sanity check on length change
                 if len(updated_summary_text) < len(current_summary_text) * 0.7 and chapter_number > 1 :
                      log.warning(f"Running summary update for Ch {chapter_number} significantly shortened the text. Check quality.")
                 # Return the new summary text wrapped in the placeholder
                 return placeholder + updated_summary_text

        except Exception as e:
            log.error(lang_conf["ERROR_UPDATING_RUNNING_SUMMARY"].format(kapitel_nummer=chapter_number, error=str(e)), exc_info=True)
            # On error, return the previous summary to avoid losing context
            return previous_summary


    def _generate_story_with_chapters(self, prompt: str, setting: str, titel: str,
                                      word_count: int, sprache: str,
                                      additional_instructions: Optional[str],
                                      max_words_per_chapter: int
                                      ) -> Optional[str]:
        """Generates a longer story by dividing it into chapters using LLM context and outline segmentation."""
        lang_conf = self._get_lang_config(sprache)
        safe_titel = self._safe_filename(titel)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        num_chapters, words_per_chapter_list = self._optimize_chapter_structure(word_count, max_words_per_chapter)
        log.info(lang_conf["INFO_GENERATING_CHAPTERS"].format(kapitel_anzahl=num_chapters))
        log.debug(f"Planned word distribution: {words_per_chapter_list}")

        # --- Setup Progress Bar ---
        self.total_steps = 1 + num_chapters * 3 + 1 # Outline(1) + N Chapters(N) + N Detail Summaries(N) + N Running Summaries(N) + Epilogue(1)
        self.current_step = 0
        self._update_progress(0) # Initialize bar

        # --- 1. Outline Generation ---
        log.info(lang_conf["INFO_GENERATING_OUTLINE"])
        plot_outline = None
        outline_final_filename = os.path.join(self.output_dir, f"outline_{safe_titel}_{timestamp}.txt")
        temp_outline_fd, temp_outline_path = tempfile.mkstemp(suffix=".txt", prefix="outline_")
        os.close(temp_outline_fd)
        outline_saved = False
        try:
            outline_prompt_context = {
                "kapitel_anzahl": num_chapters, "titel": titel, "prompt": prompt,
                "setting": setting, "wortanzahl": word_count, "zusatz_anweisungen": additional_instructions
            }
            outline_system_prompt = self._create_system_prompt(sprache, "PROMPT_OUTLINE_GEN", outline_prompt_context)
            user_prompt_outline = lang_conf["USER_PROMPT_OUTLINE"].format(kapitel_anzahl=num_chapters, titel=titel)
            log.info(lang_conf["INFO_SENDING_API_REQUEST"].format(model=self.model_name, max_tokens=min(8000, MAX_TOKENS_PER_CALL), temp=0.7) + " (Outline)")
            log.info(lang_conf["INFO_WAITING_API_RESPONSE"] + " (Outline)")
            start_time_outline = time.time()
            outline_response = self.retry_api_call(
                self.client.chat.completions.create, model=self.model_name,
                max_tokens=min(8000, MAX_TOKENS_PER_CALL), temperature=0.7,
                messages=[ {"role": "system", "content": outline_system_prompt}, {"role": "user", "content": user_prompt_outline} ]
            )
            plot_outline = outline_response.choices[0].message.content
            log.info(lang_conf["INFO_OUTLINE_CREATED"] + f" ({time.time() - start_time_outline:.1f}s)")
            self._update_progress() # Outline step done

            # Save outline (temporary for debug, final for user)
            try:
                # Save to temp file first (robustness during generation)
                with open(temp_outline_path, 'w', encoding='utf-8') as f_temp: f_temp.write(plot_outline)
                log.debug(lang_conf["INFO_SAVED_OUTLINE_TEMP"].format(dateiname=temp_outline_path))
                # Attempt to save the final outline file
                os.makedirs(os.path.dirname(outline_final_filename), exist_ok=True)
                with open(outline_final_filename, 'w', encoding='utf-8') as f_final:
                     outline_header_key = "Plot Outline for '{titel}'" if sprache == "Englisch" else "Plot-Outline für '{titel}'"
                     outline_header = f"# {outline_header_key.format(titel=titel)}\n\n"
                     f_final.write(outline_header + plot_outline)
                log.info(lang_conf["INFO_SAVED_OUTLINE_FINAL"].format(dateiname=outline_final_filename))
                outline_saved = True
            except Exception as e:
                 # Log error for final save, but continue with outline in memory/temp
                 log.error(lang_conf["ERROR_SAVING_OUTLINE_FINAL"].format(error=str(e)))
                 if os.path.exists(temp_outline_path):
                      log.info("Using outline from temporary file for generation.")
                 else: # If even temp save failed
                      log.warning(lang_conf["WARN_CANNOT_SAVE_OUTLINE_TEMP"].format(error=str(e)))

        except Exception as e:
            # Ensure progress bar finishes step even on error
            if self.current_step < 1: self._update_progress()
            log.error(f"Error creating plot outline: {str(e)}. Continuing without it.", exc_info=True)
            plot_outline = f"[{lang_conf.get('ERROR_GENERATING_OUTLINE_FALLBACK', 'Plot outline generation failed')}]"
        finally:
             # Keep temp outline file if final save failed and temp exists, otherwise delete
             if outline_saved and os.path.exists(temp_outline_path):
                 try: os.remove(temp_outline_path)
                 except Exception as e: log.warning(lang_conf["WARN_CANNOT_DELETE_TEMP"].format(error=str(e)))
             elif not outline_saved and os.path.exists(temp_outline_path):
                 log.warning(f"Keeping temporary outline file '{temp_outline_path}' as final save failed.")


        # --- 2. Generate Chapters and Contexts ---
        generated_chapters: List[str] = []
        # Tuple: (combined_context_for_prompt, raw_end_text_only)
        context_for_next_chapter: Tuple[str, str] = (lang_conf["SUMMARY_FIRST_CHAPTER"], "")
        running_plot_summary: str = lang_conf.get("RUNNING_SUMMARY_PLACEHOLDER","") + lang_conf.get("RUNNING_SUMMARY_INITIAL","")

        for i in range(num_chapters):
            chapter_number = i + 1
            target_words_chapter = words_per_chapter_list[i]
            log.info(lang_conf["INFO_GENERATING_CHAPTER_NUM"].format(kapitel_nummer=chapter_number, kapitel_anzahl=num_chapters))

            # --- A. Prepare Combined Context (Use the combined string directly) ---
            kombinierter_kontext_prompt, _ = context_for_next_chapter # Unpack, only need combined part

            # --- B. Extract Outline Segment ---
            current_outline_segment = self._extract_outline_segment(
                plot_outline, chapter_number, num_chapters, sprache
            )
            if not current_outline_segment:
                 # If segment is missing, use a clear placeholder
                 log.warning(f"Using placeholder for Chapter {chapter_number} outline segment.")
                 default_segment = "[Outline Segment nicht verfügbar]" if sprache == "Deutsch" else "[Outline Segment Unavailable]"
                 current_outline_segment = default_segment


            # --- C. Prepare Chapter Generation Context ---
            prev_chapter_num = chapter_number - 1
            chapter_context = {
                "kapitel_nummer": chapter_number, "kapitel_anzahl": num_chapters, "titel": titel,
                # "prompt": prompt, "setting": setting, # Less critical now with outline focus
                "plot_outline_segment": current_outline_segment, # Pass segment or placeholder
                "zusammenfassung_vorher": kombinierter_kontext_prompt, # Combined LLM summary + Raw End
                "running_plot_summary": running_plot_summary, # Running summary
                "kapitel_wortanzahl": target_words_chapter,
                "min_kapitel_worte": int(target_words_chapter * 0.7), # Adjusted min slightly lower
                "max_kapitel_worte": int(target_words_chapter * 1.6), # Adjusted max slightly higher
                "zusatz_anweisungen": additional_instructions,
                "prev_kapitel_nummer": prev_chapter_num,
            }

            chapter_text = None
            chapter_generation_failed = False
            try:
                # --- D. Generate Chapter ---
                chapter_text = self._generate_single_chapter_api(
                    chapter_number, num_chapters, chapter_context, sprache, titel
                )
                if chapter_text is None:
                     # If API call failed and returned None, create error placeholder
                     log.error(f"Chapter {chapter_number} generation returned None. Creating error placeholder.")
                     error_title = lang_conf["ERROR_CHAPTER_TITLE"].format(kapitel_nummer=chapter_number)
                     error_content = lang_conf["ERROR_CHAPTER_CONTENT"]
                     chapter_text = f"## {error_title}\n\n{error_content}" # Use H2 for error title
                     if chapter_number == 1: # Add H1 Main Title only for first chapter failure
                          chapter_text = f"# {titel}\n\n{chapter_text}"
                     chapter_generation_failed = True # Flag failure
                     # Do NOT append here, append after context generation block

                self._update_progress() # Chapter gen step done

                # --- E. Generate Detailed Context for NEXT chapter ---
                is_error_content = lang_conf.get("ERROR_CHAPTER_CONTENT","<ERROR>") in chapter_text
                is_rescue_notice = lang_conf.get("RESCUED_CHAPTER_NOTICE","<RESCUE>") in chapter_text

                if not is_error_content and not is_rescue_notice:
                     context_for_next_chapter = self._generate_chapter_summary_llm(chapter_text, chapter_number, sprache)
                else:
                     log.warning(f"Skipping detailed context generation after faulty/rescued chapter {chapter_number}.")
                     fallback_summary = f"{lang_conf['SUMMARY_LLM_PREFIX']}\n{lang_conf['SUMMARY_FALLBACK']}"
                     context_for_next_chapter = (fallback_summary, "") # Provide fallback, no raw end

                self._update_progress() # Detail summary step done

                # --- F. Update Running Summary ---
                if not is_error_content and not is_rescue_notice:
                     running_plot_summary = self._update_running_summary_llm(running_plot_summary, chapter_text, chapter_number, sprache)
                else:
                     log.warning(f"Skipping running summary update after faulty/rescued chapter {chapter_number}.")
                     # Keep previous running_plot_summary

                self._update_progress() # Running summary step done

                # Append the generated (or error placeholder) chapter text now
                generated_chapters.append(chapter_text)

            except Exception as e:
                # Catch unexpected errors within the loop for this chapter
                log.error(f"Critical error during processing of chapter {chapter_number}: {e}", exc_info=True)
                 # Ensure progress steps for this chapter are marked complete on loop error
                steps_done_this_loop = (self.current_step - 1) % 3 # Steps after outline
                steps_remaining_this_loop = 3 - steps_done_this_loop
                self._update_progress(steps_remaining_this_loop)
                # Create error placeholder if chapter text wasn't generated
                if chapter_text is None:
                     error_title = lang_conf["ERROR_CHAPTER_TITLE"].format(kapitel_nummer=chapter_number)
                     error_content = lang_conf["ERROR_CHAPTER_CONTENT"]
                     chapter_text = f"## {error_title}\n\n{error_content}" # Use H2
                     if chapter_number == 1: chapter_text = f"# {titel}\n\n{chapter_text}"
                     generated_chapters.append(chapter_text)
                # Decide whether to continue or abort all generation? For now, continue with error placeholder.
                # return None # Option to abort entire generation

        # --- 3. Join Chapters ---
        full_story = self._join_chapters(generated_chapters, titel, sprache)
        if not full_story:
             log.error("Failed to join chapters.")
             if self.current_step < self.total_steps: self._update_progress(self.total_steps - self.current_step)
             return None

        # --- 4. Generate Epilogue ---
        epilogue_generated = False
        if generated_chapters:
            last_chapter_text = generated_chapters[-1]
            target_words_last_chapter = words_per_chapter_list[-1] if words_per_chapter_list else 0
            actual_words_last = len(last_chapter_text.split())
            is_rescued = lang_conf.get("RESCUED_CHAPTER_NOTICE", "<RESCUE>") in last_chapter_text
            is_short = target_words_last_chapter > 0 and actual_words_last < target_words_last_chapter * 0.7
            is_error_content = lang_conf.get("ERROR_CHAPTER_CONTENT","<ERROR>") in last_chapter_text

            # Generate epilogue unless the very last chapter was a total generation failure
            should_generate_epilogue = not is_error_content

            if should_generate_epilogue:
                if is_short or is_rescued:
                     log.info(lang_conf["INFO_LAST_CHAPTER_INCOMPLETE"] + f" (Short: {is_short}, Rescued: {is_rescued})")
                else:
                     log.info("Proceeding to generate epilogue for final conclusion.")

                # --- Prepare Context for Epilogue ---
                # Use the context generated *after* the last chapter completed
                kombinierter_kontext_epilog, raw_end_part_epilog = context_for_next_chapter

                # Ensure the raw end part from the context tuple is used, not the full last chapter text again
                epilogue_raw_end_context = raw_end_part_epilog if raw_end_part_epilog else last_chapter_text[-500:] # Fallback

                epilogue_text = self._generate_epilogue(
                    titel, plot_outline or f"[{lang_conf.get('ERROR_GENERATING_OUTLINE_FALLBACK', 'Outline Unavailable')}]", # Pass full outline or placeholder
                    kombinierter_kontext_epilog, # Detailed context from last chapter (Summary + Raw End Markers)
                    running_plot_summary, # Running summary of whole story
                    epilogue_raw_end_context, # Pass only the relevant raw end snippet
                    sprache, additional_instructions,
                    num_chapters # Pass chapter count for prompt context
                )
                if epilogue_text:
                    # Append epilogue cleanly, ensuring H2 formatting is likely handled by _generate_epilogue
                    full_story = full_story.strip() + "\n\n" + epilogue_text.strip()
                    epilogue_generated = True
                else:
                    log.warning("Epilogue generation failed.")
                    # Append notice if epilogue failed
                    full_story += f"\n\n{lang_conf['RESCUED_EPILOG_NOTICE']}"

            else:
                 log.warning("Skipping epilogue generation because the last chapter failed.")
                 full_story += f"\n\n{lang_conf['RESCUED_EPILOG_NOTICE']}" # Add notice if skipped due to error

        else: # No chapters generated at all
             log.error("Cannot generate epilogue as no chapters were generated.")

        self._update_progress() # Epilogue step complete (or skipped)

        # Final cleanup & return
        full_story = self._clean_text_ending(full_story.strip(), sprache)
        # Ensure progress bar hits 100% if any steps were skipped due to errors
        if self.current_step < self.total_steps:
            self._update_progress(self.total_steps - self.current_step)
        return full_story


    def _generate_single_chapter_api(self, chapter_number: int, total_chapters: int,
                                     chapter_context: Dict[str, Any],
                                     sprache: str, titel: str
                                     ) -> Optional[str]:
        """Generates a single chapter via the API with quality focus and hierarchical context."""
        lang_conf = self._get_lang_config(sprache)
        temp_chapter_fd, temp_chapter_path = tempfile.mkstemp(suffix=".txt", prefix=f"chapter_{chapter_number}_")
        os.close(temp_chapter_fd)

        try:
            target_words = chapter_context["kapitel_wortanzahl"]
            # Slightly increased token buffer for complex hierarchical prompt
            max_tokens = min(int(target_words * TOKEN_WORD_RATIO * 1.4) + 600, MAX_TOKENS_PER_CALL)
            temperature = 0.75 # Keep temperature moderate

            # Create system prompt using the full context dictionary (includes segment, summaries etc.)
            system_prompt = self._create_system_prompt(sprache, "PROMPT_CHAPTER_GEN", chapter_context)
            user_prompt = lang_conf["USER_PROMPT_CHAPTER"].format(kapitel_nummer=chapter_number, titel=titel)

            log.info(lang_conf["INFO_SENDING_API_REQUEST"].format(model=self.model_name, max_tokens=max_tokens, temp=temperature) + f" (Chapter {chapter_number})")
            log.info(lang_conf["INFO_WAITING_API_RESPONSE"] + f" (Chapter {chapter_number})")
            start_time = time.time()

            response = self.retry_api_call(
                self.client.chat.completions.create,
                model=self.model_name, max_tokens=max_tokens, temperature=temperature,
                messages=[ {"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt} ]
            )

            chapter_text = response.choices[0].message.content
            duration = time.time() - start_time
            log.info(lang_conf["INFO_CHAPTER_COMPLETE"].format(kapitel_nummer=chapter_number, dauer=duration))

            # Save temporarily for potential rescue
            try:
                with open(temp_chapter_path, 'w', encoding='utf-8') as f: f.write(chapter_text)
                log.debug(lang_conf["INFO_SAVED_TEMP_FILE"].format(dateiname=temp_chapter_path))
            except Exception as e: log.warning(lang_conf["WARN_CANNOT_WRITE_TEMP"].format(error=str(e)))

            # Basic validation (word count check)
            word_c = len(chapter_text.split())
            min_target = chapter_context.get("min_kapitel_worte", 50)
            max_target = chapter_context.get("max_kapitel_worte", target_words * 2) # Wider tolerance
            if word_c < min_target * 0.5: # Check if significantly shorter than min
                 log.warning(f"Chapter {chapter_number} is very short ({word_c} words, min target {min_target}).")
            elif word_c > max_target * 1.2: # Check if significantly longer than max
                 log.warning(f"Chapter {chapter_number} is very long ({word_c} words, max target {max_target}).")

            # Formatting and cleanup (ensure H2 chapter heading)
            chapter_text = self._format_chapter(chapter_number, chapter_text, titel, sprache)
            chapter_text = self._clean_text_ending(chapter_text, sprache)
            return chapter_text

        except Exception as e:
            # Log API errors specifically if they haven't been logged by retry_api_call already
            if not isinstance(e, (OpenAI.APIError, Exception)) or "All API retries failed" not in str(e):
                 log.error(f"Unexpected error processing chapter {chapter_number} after API call: {e}", exc_info=True)
            # Log the general chapter error message
            log.error(lang_conf["ERROR_API_REQUEST_CHAPTER"].format(kapitel_nummer=chapter_number, error=str(e)), exc_info=False) # Don't need full trace here usually

            # Attempt rescue from temp file
            if os.path.exists(temp_chapter_path):
                try:
                    with open(temp_chapter_path, 'r', encoding='utf-8') as f:
                        partial_chapter = f.read()
                    if len(partial_chapter.strip()) > MIN_CHARS_FOR_RESCUE:
                        log.info(lang_conf["INFO_RESCUED_PARTIAL_CONTENT"].format(chars=len(partial_chapter)))
                        partial_chapter = self._format_chapter(chapter_number, partial_chapter, titel, sprache)
                        partial_chapter = self._clean_text_ending(partial_chapter, sprache)
                        # Append the rescue notice
                        return partial_chapter + f"\n\n{lang_conf['RESCUED_CHAPTER_NOTICE']}"
                except Exception as e2:
                    log.error(lang_conf["ERROR_READING_TEMP_FILE"].format(error=str(e2)))

            # If rescue fails or temp file not useful, return None to signal failure
            return None
        finally:
            # Clean up temp file
            if os.path.exists(temp_chapter_path):
                try: os.remove(temp_chapter_path)
                except Exception as e: log.warning(lang_conf["WARN_CANNOT_DELETE_TEMP"].format(error=str(e)))


    def _format_chapter(self, chapter_number: int, chapter_text: str, titel: str, sprache: str) -> str:
        """Ensures the chapter is correctly formatted with H1 Title (Ch1) and H2 Chapters."""
        lang_conf = self._get_lang_config(sprache)
        chapter_text = chapter_text.strip()

        main_title_prefix = f"# {titel}" # H1
        if sprache == "Deutsch":
             chapter_word_base = "Kapitel"
             default_chapter_title_base = lang_conf.get("DEFAULT_CHAPTER_TITLE", "Kapitel {kapitel_nummer}")
             first_chapter_title_base = lang_conf.get("FIRST_CHAPTER_TITLE", "Kapitel {kapitel_nummer}")
        else: # Englisch
             chapter_word_base = "Chapter"
             default_chapter_title_base = lang_conf.get("DEFAULT_CHAPTER_TITLE", "Chapter {kapitel_nummer}")
             first_chapter_title_base = lang_conf.get("FIRST_CHAPTER_TITLE", "Chapter {kapitel_nummer}")

        # Format the base titles with the actual chapter number
        default_chapter_title = default_chapter_title_base.format(kapitel_nummer=chapter_number)
        first_chapter_title = first_chapter_title_base.format(kapitel_nummer=1) # Always format Ch 1 with 1

        chapter_heading_correct_prefix = f"## {default_chapter_title}" # Target H2 heading structure
        first_chapter_heading_correct_prefix = f"## {first_chapter_title}" # Target H2 heading structure for Ch1
        chapter_prefix_pattern = rf"^\s*#+\s*{chapter_word_base}\s+{chapter_number}\b.*" # Match any # level, ensure chapter number matches

        lines = chapter_text.split('\n')
        # Remove leading empty lines for cleaner processing
        while lines and not lines[0].strip(): lines.pop(0)

        if chapter_number == 1:
            # --- Part 1: Ensure Main Title (H1) ---
            if not lines or not lines[0].strip().startswith(main_title_prefix):
                 # Remove any incorrect leading headings
                 while lines and lines[0].strip().startswith("#"):
                      log.debug(f"Ch1 Format: Removing incorrect leading heading: '{lines[0].strip()}'")
                      lines.pop(0)
                 log.debug(f"Ch1 Format: Prepending main title (H1): '{main_title_prefix}'")
                 lines.insert(0, main_title_prefix)
                 # Ensure a blank line after H1 if content follows immediately
                 if len(lines) > 1 and lines[1].strip():
                      lines.insert(1, "")
            # If main title exists, ensure it's exactly H1
            elif lines[0].strip() != main_title_prefix:
                 if re.match(r"^#+\s*" + re.escape(titel), lines[0].strip()):
                      log.debug(f"Ch1 Format: Correcting main title heading level to H1.")
                      lines[0] = main_title_prefix
                 # else: some other text, leave it for now

            # --- Part 2: Ensure Chapter 1 Heading (H2) ---
            # Find where the actual content starts (after H1 and potential blank line)
            content_start_index = 0
            if lines and lines[0].strip() == main_title_prefix:
                 content_start_index = 1
                 if len(lines) > 1 and not lines[1].strip(): # Skip blank line
                      content_start_index = 2

            has_ch1_heading = False
            if len(lines) > content_start_index:
                 # Use stricter pattern to match "Chapter 1" or "Kapitel 1"
                 ch1_heading_match = re.match(rf"^[#\s]*{chapter_word_base}\s+1\b.*", lines[content_start_index].strip(), re.IGNORECASE)
                 if ch1_heading_match:
                     has_ch1_heading = True
                     heading_text = lines[content_start_index].strip()
                     # Extract specific title part (text after "Chapter 1:")
                     title_match = re.match(rf"[#\s]*(?:{chapter_word_base})\s+1\s*:?\s*(.*)", heading_text, re.IGNORECASE)
                     specific_title_part = title_match.group(1).strip() if title_match and title_match.group(1) else ""

                     # Construct the correct H2 heading
                     # If the LLM provided a specific title part, use it after the base
                     if specific_title_part and specific_title_part.lower() != first_chapter_title.lower():
                          correct_ch1_heading = f"## {first_chapter_title}: {specific_title_part}"
                     else: # Otherwise, use the default formatted H2 heading
                          correct_ch1_heading = first_chapter_heading_correct_prefix

                     # Update the line if it's not exactly the correct H2 format
                     if heading_text != correct_ch1_heading:
                          log.debug(f"Ch1 Format: Correcting existing Ch1 heading to: '{correct_ch1_heading}'")
                          lines[content_start_index] = correct_ch1_heading
                 # else: it's content, not a Ch1 heading

            # If no Ch1 heading was found after H1
            if not has_ch1_heading:
                 correct_ch1_heading = first_chapter_heading_correct_prefix # Use default H2
                 log.debug(f"Ch1 Format: Adding default Chapter 1 heading (H2): '{correct_ch1_heading}'")
                 lines.insert(content_start_index, correct_ch1_heading)
                 # Ensure blank line after added heading if content follows immediately
                 if len(lines) > content_start_index + 1 and lines[content_start_index+1].strip():
                      lines.insert(content_start_index + 1, "")

        # Chapters > 1
        else:
            has_correct_heading = False
            if lines:
                 # Check if the first line is a heading matching the current chapter number
                 heading_match = re.match(chapter_prefix_pattern, lines[0].strip(), re.IGNORECASE)
                 if heading_match:
                     has_correct_heading = True
                     heading_text = lines[0].strip()
                     # Extract specific title part
                     title_match = re.match(rf"[#\s]*(?:{chapter_word_base})\s+{chapter_number}\s*:?\s*(.*)", heading_text, re.IGNORECASE)
                     specific_title_part = title_match.group(1).strip() if title_match and title_match.group(1) else ""

                     # Construct the correct H2 heading
                     if specific_title_part and specific_title_part.lower() != default_chapter_title.lower():
                         correct_heading = f"## {default_chapter_title}: {specific_title_part}"
                     else:
                         correct_heading = chapter_heading_correct_prefix # Use default H2

                     # Update if not exactly correct H2 format
                     if heading_text != correct_heading:
                          log.debug(f"Ch{chapter_number} Format: Correcting existing heading to: '{correct_heading}'")
                          lines[0] = correct_heading
                 # else: first line is content, heading is missing

            # If no correct heading found at the start
            if not has_correct_heading:
                 correct_heading = chapter_heading_correct_prefix # Use default H2
                 log.debug(f"Ch{chapter_number} Format: Adding default heading (H2): '{correct_heading}'")
                 # Remove incorrect heading if present
                 if lines and lines[0].strip().startswith("#"):
                      log.debug(f"Ch{chapter_number} Format: Removing incorrect leading heading: '{lines[0].strip()}'")
                      lines.pop(0)
                 lines.insert(0, correct_heading)
                 # Ensure blank line after added heading
                 if len(lines) > 1 and lines[1].strip():
                      lines.insert(1, "")

        return "\n".join(lines).strip()


    def _join_chapters(self, chapter_texts: List[str], titel: str, sprache: str) -> Optional[str]:
        """Joins the generated chapter texts into a single story string."""
        if not chapter_texts:
            log.error(f"Error: No chapter texts provided to join for '{titel}'.")
            return None # Return None to indicate failure

        # First chapter should already have H1 title and H2 chapter heading
        full_story = chapter_texts[0].strip()

        # Append the rest, ensuring double newline separation
        for i in range(1, len(chapter_texts)):
            full_story += "\n\n" + chapter_texts[i].strip()

        return full_story


    def _generate_epilogue(self, titel: str, plot_outline: str, combined_context_last_chapter: str,
                           running_summary: str, # Added running summary
                           raw_end_snippet_last_chapter: str, # Use only the snippet here
                           sprache: str,
                           additional_instructions: Optional[str],
                           last_chapter_number: int # Added for prompt context
                           ) -> Optional[str]:
        """Generates an epilogue using combined context, running summary, and raw end snippet."""
        lang_conf = self._get_lang_config(sprache)
        temp_epilogue_fd, temp_epilogue_path = tempfile.mkstemp(suffix=".txt", prefix="epilogue_")
        os.close(temp_epilogue_fd)

        log.info(lang_conf["INFO_GENERATING_EPILOG"])
        try:
            epilogue_context = {
                "titel": titel,
                "plot_outline": plot_outline, # Full outline for context
                "zusammenfassung_vorher": combined_context_last_chapter, # LLM Summary + Raw End Markers of last chapter
                "running_plot_summary": running_summary, # Overall plot summary
                "letztes_kapitel_ende": raw_end_snippet_last_chapter, # Pass only the snippet
                "zusatz_anweisungen": additional_instructions,
                "kapitel_anzahl": last_chapter_number # Number of the last *actual* chapter
            }
            system_prompt = self._create_system_prompt(sprache, "PROMPT_EPILOG_GEN", epilogue_context)
            user_prompt = lang_conf["USER_PROMPT_EPILOG"].format(titel=titel)

            # Increased token limit for epilogue to allow proper resolution
            max_tokens_epilogue = min(3500, MAX_TOKENS_PER_CALL)
            log.info(lang_conf["INFO_SENDING_API_REQUEST"].format(model=self.model_name, max_tokens=max_tokens_epilogue, temp=0.7) + " (Epilogue)")
            log.info(lang_conf["INFO_WAITING_API_RESPONSE"] + " (Epilogue)")
            start_time = time.time()

            response = self.retry_api_call(
                self.client.chat.completions.create,
                model=self.model_name,
                max_tokens=max_tokens_epilogue,
                temperature=0.7, # Keep temperature moderate for conclusion
                messages=[ {"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt} ]
            )
            epilogue_text = response.choices[0].message.content
            duration = time.time() - start_time
            log.info(lang_conf["INFO_EPILOG_GENERATED"].format(dauer=duration))

            # Save temp file for potential rescue
            try:
                with open(temp_epilogue_path, 'w', encoding='utf-8') as f: f.write(epilogue_text)
                log.debug(lang_conf["INFO_SAVED_TEMP_FILE"].format(dateiname=temp_epilogue_path))
            except Exception as e: log.warning(lang_conf["WARN_CANNOT_WRITE_TEMP"].format(error=str(e)))

            # Format and clean (ensure H2 heading for Epilogue)
            epilogue_text = epilogue_text.strip()
            epilog_title_base = lang_conf.get('EPILOG_TITLE', 'Epilog' if sprache == 'Deutsch' else 'Epilogue')
            epilogue_title_correct = f"## {epilog_title_base}" # Target H2

            if not epilogue_text.startswith("##"): # Check if it starts with H2
                 lines = epilogue_text.split('\n')
                 # Remove any incorrect leading headings
                 while lines and lines[0].strip().startswith('#'):
                      log.debug(f"Epilogue Format: Removing incorrect leading line: {lines[0].strip()}")
                      lines.pop(0)
                 epilogue_text = "\n".join(lines).strip()
                 log.debug(f"Epilogue Format: Prepending heading '{epilogue_title_correct}'")
                 epilogue_text = f"{epilogue_title_correct}\n\n{epilogue_text}"
            else: # Starts with ##, verify/correct the title text
                 lines = epilogue_text.split('\n')
                 heading_text = lines[0].strip()
                 # Try to extract potential custom title after ##
                 title_match = re.match(r"##\s*(.*)", heading_text)
                 specific_title = title_match.group(1).strip() if title_match else epilog_title_base

                 # If the specific title is the main story title, replace with default epilogue title
                 if specific_title.lower() == titel.lower():
                     correct_heading = epilogue_title_correct
                 # If it's different from the default base, assume LLM gave a specific title
                 elif specific_title.lower() != epilog_title_base.lower():
                      correct_heading = f"## {specific_title}" # Keep LLM's specific title, ensure H2
                 else: # It matches the default base, ensure H2 format
                      correct_heading = epilogue_title_correct

                 if heading_text != correct_heading:
                      log.debug(f"Epilogue Format: Correcting heading to '{correct_heading}'")
                      lines[0] = correct_heading
                 epilogue_text = "\n".join(lines)

            epilogue_text = self._clean_text_ending(epilogue_text, sprache)
            return epilogue_text

        except Exception as e:
             # Log API errors specifically if not already logged by retry
             if not isinstance(e, (OpenAI.APIError, Exception)) or "All API retries failed" not in str(e):
                  log.error(f"Unexpected error during epilogue generation: {e}", exc_info=True)
             log.error(lang_conf["ERROR_GENERATING_EPILOG"].format(error=str(e)), exc_info=False)

             # Attempt rescue from temp file
             if os.path.exists(temp_epilogue_path):
                 try:
                     with open(temp_epilogue_path, 'r', encoding='utf-8') as f:
                         partial_epilogue = f.read()
                     if len(partial_epilogue.strip()) > MIN_CHARS_FOR_RESCUE:
                         log.info(lang_conf["INFO_RESCUED_PARTIAL_CONTENT"].format(chars=len(partial_epilogue)))
                         partial_epilogue = self._clean_text_ending(partial_epilogue.strip(), sprache)
                         # Format the rescued part with H2 title
                         epilog_title_base = lang_conf.get('EPILOG_TITLE', 'Epilog' if sprache == 'Deutsch' else 'Epilogue')
                         epilogue_title_correct = f"## {epilog_title_base}"
                         if not partial_epilogue.startswith(epilogue_title_correct):
                              partial_epilogue = f"{epilogue_title_correct}\n\n{partial_epilogue}"
                         # Add the rescue notice
                         return partial_epilogue + f"\n\n{lang_conf['RESCUED_EPILOG_NOTICE']}"
                 except Exception as e2:
                     log.error(lang_conf["ERROR_READING_TEMP_FILE"].format(error=str(e2)))
             # Return None if epilogue generation failed and rescue didn't work
             return None
        finally:
            # Delete temporary epilogue file
            if os.path.exists(temp_epilogue_path):
                try: os.remove(temp_epilogue_path)
                except Exception as e: log.warning(lang_conf["WARN_CANNOT_DELETE_TEMP"].format(error=str(e)))


    # --- Main Generate Method (Entry Point) ---
    def generate(self, prompt: str, setting: str, titel: str,
                 word_count: int = DEFAULT_TARGET_WORDS,
                 sprache: str = "Deutsch",
                 additional_instructions: Optional[str] = None,
                 chapter_mode: bool = True,
                 max_words_per_chapter: int = MAX_WORDS_PER_CHAPTER
                 ) -> Optional[str]:
        """
        Main method: Generates a story, deciding on chapter mode based on word count.
        Uses enhanced LLM-based context passing (detailed + running) between chapters if applicable.
        Returns the story as a string or None on failure. CLI version integrates progress bar.
        """
        start_time_total = time.time()
        try:
            lang_conf = self._get_lang_config(sprache)
        except ValueError as e:
            log.error(f"Configuration error: {e}")
            return None # Cannot proceed without valid language config

        # Use buffer factor primarily for deciding chapter mode
        target_word_count_with_buffer = int(word_count * WORD_COUNT_BUFFER_FACTOR)
        log.info(f"Requested word count: {word_count}, Target with {WORD_COUNT_BUFFER_FACTOR:.1f}x buffer: {target_word_count_with_buffer}")

        # Apply min/max constraints based on buffered count
        if target_word_count_with_buffer < MIN_STORY_WORDS:
            log.warning(lang_conf["WARN_LOW_WORDCOUNT"].format(
                wortanzahl=target_word_count_with_buffer, min_val=MIN_STORY_WORDS))
            target_word_count_with_buffer = MIN_STORY_WORDS
        elif target_word_count_with_buffer > MAX_STORY_WORDS_NO_CHAPTERS and not chapter_mode:
            log.warning(lang_conf["WARN_HIGH_WORDCOUNT"].format(
                wortanzahl=target_word_count_with_buffer, max_val=MAX_STORY_WORDS_NO_CHAPTERS))
            target_word_count_with_buffer = MAX_STORY_WORDS_NO_CHAPTERS

        log.info(lang_conf["INFO_GENERATING_STORY"].format(
            titel=titel, wortanzahl=target_word_count_with_buffer, sprache=sprache
        ))

        # Decide chapter mode based on buffered word count and API limits
        estimated_input_tokens = 1500 # Generous estimate for base prompt parts
        max_output_tokens_single_call = MAX_TOKENS_PER_CALL - estimated_input_tokens
        effective_max_words_single_call = max(0, max_output_tokens_single_call / TOKEN_WORD_RATIO)

        # Use chapter mode if explicitly enabled AND
        # (buffered count > max per chapter OR buffered count > effective max for single call)
        use_chapter_mode = chapter_mode and (
            target_word_count_with_buffer > max_words_per_chapter or
            target_word_count_with_buffer > effective_max_words_single_call
        )

        story = None
        try:
            if use_chapter_mode:
                 log.info(lang_conf["INFO_CHAPTER_MODE_ACTIVATED"].format(wortanzahl=target_word_count_with_buffer))
                 story = self._generate_story_with_chapters(
                     prompt, setting, titel, target_word_count_with_buffer, sprache,
                     additional_instructions, max_words_per_chapter
                 )
            else:
                # Double-check feasibility before attempting single call
                if target_word_count_with_buffer > effective_max_words_single_call:
                     log.error(f"Target word count {target_word_count_with_buffer} is too high for single call generation (estimated max: {effective_max_words_single_call:.0f} words with MAX_TOKENS_PER_CALL={MAX_TOKENS_PER_CALL}). Chapter mode is required but was disabled or not triggered.")
                     return None # Fail because single call is impossible

                log.info("Generating story as a single segment.")
                # Setup progress bar for single step
                self.total_steps = 1
                self.current_step = 0
                story = self._generate_single_story(
                    prompt, setting, titel, target_word_count_with_buffer, sprache, additional_instructions
                )

            # --- Final Result Handling ---
            if story is None:
                # Error logged within the called function
                log.error(lang_conf["ERROR_GENERATION_FAILED"] + f" (Title: '{titel}')")
                return None

            actual_word_count = len(story.split())
            total_duration = time.time() - start_time_total
            log.info(lang_conf["INFO_FINAL_WORD_COUNT"].format(wortanzahl=actual_word_count) + f" (Total time: {total_duration:.1f}s)")
            return story

        except Exception as e:
             # Catch any unexpected errors during the generate process itself
             log.error(f"An unexpected critical error occurred during story generation orchestrator: {e}", exc_info=True)
             # Ensure progress bar finishes if it was started
             if self.total_steps > 0 and self.current_step < self.total_steps:
                  self._update_progress(self.total_steps - self.current_step)
             return None

    def save_as_text_file(self, content: str, title: str, language: str, output_path: Optional[str] = None) -> str:
        """Saves the content as a text file. (Kept from original CLI script)"""
        lang_conf = self._get_lang_config(language)
        safe_title = self._safe_filename(title)
        filename = ""
        output_dir_final = output_path if output_path else self.output_dir # Use provided path or instance default

        # Default filename components
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"story_{safe_title}_{timestamp}.txt" if language == "Englisch" else f"geschichte_{safe_title}_{timestamp}.txt"

        # Determine final filename/path
        if output_dir_final:
             # Check if output_dir_final looks like a directory or a specific file path
             path_isdir = os.path.isdir(output_dir_final)
             path_has_extension = os.path.splitext(output_dir_final)[1].lower() == ".txt"

             if path_isdir or not os.path.splitext(output_dir_final)[1]: # Treat as dir if it exists as dir or has no extension
                  os.makedirs(output_dir_final, exist_ok=True)
                  filename = os.path.join(output_dir_final, base_filename)
             elif path_has_extension: # Treat as a specific filename
                  filename = output_dir_final
                  # Ensure the directory for the specified file exists
                  os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
             else: # Path exists but is not a dir and has no .txt extension - treat as dir + base filename
                  os.makedirs(output_dir_final, exist_ok=True)
                  filename = os.path.join(output_dir_final, base_filename)
        else:
            # Default: save in the current directory (should usually be covered by output_dir)
            filename = base_filename

        # Format content (ensure H1 title is at the start) - relies on content already being formatted correctly by generate methods
        # formatted_content = self._format_story(content, title, language) # Formatting should be done, just ensure clean ending
        formatted_content = content.strip() + "\n" # Ensure single newline at end

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(formatted_content)
            log.info(lang_conf["INFO_SAVED_TEXT_FILE"].format(dateiname=filename)) # 'dateiname' key
            return filename # Return the actual saved path
        except Exception as e:
             log.error(f"Error saving text file '{filename}': {str(e)}")
             raise # Re-raise the exception to signal failure


# === Main Part / Command Line Interface (Adapted) ===
def main():
    parser = argparse.ArgumentParser(
        description="Generates short stories using LLMs with enhanced quality control and chapter handling.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--prompt", type=str, required=True, help="Prompt/Basic idea for the story")
    parser.add_argument("--setting", type=str, required=True, help="Setting (Place, Time, Atmosphere)")
    parser.add_argument("--title", type=str, required=True, help="Title of the story") # Kept original name
    parser.add_argument("--wordcount", type=int, default=DEFAULT_TARGET_WORDS, # Kept original name
                        help=f"Approximate TARGET word count (buffer factor {WORD_COUNT_BUFFER_FACTOR:.1f}x applied internally)")
    parser.add_argument("--language", type=str, default="Deutsch", choices=SUPPORTED_LANGUAGES, # Kept original name
                        help="Language of the story")
    parser.add_argument("--additional", type=str, help="Additional instructions for the generation (e.g., style, character notes)") # Kept original name
    parser.add_argument("--api-key", type=str, help="Nebius API Key (alternatively use NEBIUS_API_KEY env var)")
    parser.add_argument("--model", type=str, default=MODELL_NAME, help="Name of the LLM model to use")
    parser.add_argument("--save-text", action="store_true", help="Save the generated story as a text file")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Directory for output files (story text, outline). Can also be a full path ending in .txt for the story file.")
    parser.add_argument("--max-words-per-chapter", type=int, default=MAX_WORDS_PER_CHAPTER, # Kept original name
                        help="Target maximum word count per chapter generation call (influences chapter mode trigger)")
    parser.add_argument("--no-chapter-mode", action="store_true", # Kept original name
                        help=f"Force single-segment generation (only feasible for word counts < ~{int(MAX_TOKENS_PER_CALL / TOKEN_WORD_RATIO)} words)")
    parser.add_argument("--debug", action="store_true", help="Enable detailed debug logging")

    args = parser.parse_args()

    # Configure Logging Level based on --debug flag
    log_level = logging.DEBUG if args.debug else logging.INFO
    # Force reconfiguration to apply the new level
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s', force=True)
    # Suppress overly verbose logs from http client library used by OpenAI
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    log.info(f"Log level set to {'DEBUG' if args.debug else 'INFO'}.")

    # Create output directory if specified as a directory and doesn't exist
    output_dir_path = args.output_dir
    if not os.path.splitext(output_dir_path)[1]: # If it doesn't have an extension, assume it's a directory
        try:
             if output_dir_path and output_dir_path != ".":
                  os.makedirs(output_dir_path, exist_ok=True)
                  log.info(f"Output directory ensured: {output_dir_path}")
        except Exception as e:
             log.error(f"Error creating output directory '{output_dir_path}': {e}")
             sys.exit(1)
    elif os.path.dirname(output_dir_path): # If it's a file path, ensure its directory exists
         try:
              os.makedirs(os.path.dirname(output_dir_path), exist_ok=True)
              log.info(f"Output directory for file ensured: {os.path.dirname(output_dir_path)}")
         except Exception as e:
              log.error(f"Error creating output directory '{os.path.dirname(output_dir_path)}' for file: {e}")
              sys.exit(1)


    # --- Instantiate Generator and Run ---
    generator = None # Initialize generator to None
    try:
        # Pass output_dir to generator for potential use (like saving outline)
        generator = StoryGenerator(api_key=args.api_key, model=args.model, output_dir=args.output_dir)
        lang_conf = generator._get_lang_config(args.language) # Use method after init

        # Generate the story using the main generate method
        story = generator.generate(
            prompt=args.prompt,
            setting=args.setting,
            titel=args.title,
            word_count=args.wordcount, # Pass the user's target word count
            sprache=args.language,
            additional_instructions=args.additional,
            chapter_mode=not args.no_chapter_mode,
            max_words_per_chapter=args.max_words_per_chapter
            # Progress bar is handled internally now via show_progress_bar
        )

        # Check if story generation was successful
        if not story:
            # Error should have been logged already
            sys.exit(1) # Exit with error code

        # --- Output / Save ---
        if args.save_text:
            try:
                 # Use the save_as_text_file method, passing the output_dir argument
                 generator.save_as_text_file(
                     content=story, title=args.title, language=args.language,
                     output_path=args.output_dir # Pass the arg directly
                 )
            except Exception as e:
                 log.error(f"Could not save story to text file: {e}", exc_info=True)
                 # Fallback: Print to console if saving fails
                 print("\n--- Story (Could not be saved to file) ---")
                 print(story)
                 print("--- End Story ---")
        else:
            # Print story to console if not saving
            print("\n" + "=" * 80)
            print(story) # Story content already includes titles/headings
            print("=" * 80)


    except ValueError as ve:
         # Catch configuration errors (e.g., unsupported language, missing API key)
         log.error(f"Configuration Error: {ve}")
         sys.exit(1)
    except Exception as e:
        # Catch any other unexpected errors during execution
        log.error(f"An unexpected critical error occurred in main: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()