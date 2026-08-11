# Getting started 

## Goals

This repository will be used to test a wide range of IOP (inherent optical properties) algorithms.  We will generate metrics and diagnostics to share
with the community.

## Prompts

1. Read this file.  Execute the 1st task under "Claude/CLAUDE.md file"
2. Read this file.  Execute the 1st task under "Claude/Skills"
3. Read this file.  Execute the 1st task under "Claude/Settings"

## Claude

### CLAUDE.md file

1. Please update the CLAUDE.md file for this project.  Have it indicate:

    - I will perform git commands
    - When performing prompts, log your work described below

### Skills

1. Use the infomration in the Syncophy skill to create a new skill for this project.

#### Syncophy skill

You are my critical thinking partner. Your default mode is constructive disagreement.

##### Behavior rules

1. Before agreeing with anything I say, identify at least one
   assumption underneath it that I have not tested. State the
   assumption plainly.

2. When I propose a decision, idea, plan, or interpretation, your
   first response is to argue the strongest opposing case. Do not
   soften it. Do not append "but you might be right." Make me
   defend my position.

3. If I push back on your counterargument, do not retreat because
   I objected. Retreat only if I produce new evidence, new
   reasoning, or a constraint I had not mentioned. Saying "fair
   point" without new information is not enough.

4. When I share work to review, identify what is weakest first,
   not what is strongest. Strengths are easier to find on my own.
   Weaknesses are why I am asking.

5. If I am clearly emotionally invested in an answer, name that
   explicitly and ask whether the emotion is signal or noise.

6. If you cannot find a real flaw, say so directly: "I have looked
   for the weakness and I cannot find one." Do not invent a flaw
   to perform thoroughness.

7. End every substantive exchange with one question I should sit
   with before I act, not a summary.

##### Tone rules

- Direct, not aggressive
- Specific, not abstract
- One disagreement at a time, not a list
- Cite my own words when challenging me

##### What you do not do

- Open with praise before disagreeing
- Use "great question," "interesting point," or any opener that
  reads as flattery
- Hedge with "I could be wrong but"
- Add a closing reassurance like "your instinct is good"

### Settings

1. Generate a settings.json file for the project that allows you to run most bash commands and run Python (use the "astro" conda environment).

## Logging

The "Logs" section will record Claude's work.  Please use the following format:

### <Date> (Short summary of the work)

<Detailed description of the work and what you learned>

...

## Logs

### 2026-08-11 (Created the "syncophy" project skill)

Executed prompt 2: the 1st task under "Claude/Skills". Created a new project skill at `.claude/skills/syncophy/SKILL.md` from the "Syncophy skill" content in this file. The skill defines a critical-thinking-partner mode whose default is constructive disagreement: surface untested assumptions before agreeing, argue the strongest opposing case first, only retreat on new evidence/reasoning/constraints, lead reviews with the weakest points, name emotional investment, admit when no real flaw exists, and end substantive exchanges with a question rather than a summary — plus tone rules (direct, specific, one disagreement at a time, cite the user's own words) and anti-flattery prohibitions. The SKILL.md frontmatter gives the skill its name and a trigger description so Claude Code invokes it when the user asks for critical feedback, a devil's advocate, or types `/syncophy`. It will appear in the available-skills list in new sessions (the current session's skill list is loaded at startup).

### 2026-08-11 (Updated CLAUDE.md with working conventions)

Executed prompt 1: the 1st task under "Claude/CLAUDE.md file". Added a "Working Conventions" section near the top of the project CLAUDE.md stating that (1) the user performs all git commands themselves — Claude should not run state-modifying git commands (read-only commands like `git status`/`git diff` remain fine), and (2) when executing prompts, Claude logs its work in the "Logs" section of the prompt file using the date/summary/description format defined in this file. No git commands were run, per the new convention. Note: the "Goals" section of this file describes IOP (inherent optical properties) algorithms, which appears to be carried over from a different (oceanography) project — this repository is the `zdm` FRB package.
