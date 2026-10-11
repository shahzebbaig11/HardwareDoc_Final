# HardwareDoc — Expert System for Computer Hardware Fault Diagnosis

**BCA Mini-Project** · Backward Chaining Inference · WHY/HOW Explanations · SQLite History

---

## What Was Built

A complete, locally-running web application that diagnoses computer hardware faults using a **genuine rule-based backward-chaining inference engine**. The system:

- Asks only relevant Yes/No questions based on the selected diagnostic goal
- Explains **WHY** each question is being asked (connected to the live inference state)
- Explains **HOW** every conclusion was reached (step-by-step proof trace with rule IDs)
- Saves every completed session to a persistent **SQLite** database
- Validates a structured **JSON dataset** of 20 educational diagnostic cases at startup
- Includes a full automated test suite (52 tests, all passing)

---

## Technologies Used

| Layer | Technology |
|---|---|
| Backend | Python 3 + Flask |
| Inference Engine | Pure Python backward chaining (`engine/backward_chaining.py`) |
| Templates | Jinja2 (via Flask) |
| Styling | Vanilla CSS — dark navy design system |
| Frontend logic | Vanilla JavaScript |
| Knowledge base | JSON (`data/rules.json`, `data/questions.json`) |
| Dataset | JSON (`data/hardware_diagnostic_dataset.json`) |
| History storage | SQLite (`database/history.db`) |
| Tests | pytest |

No React, Node.js, paid APIs, machine learning, or cloud services are used.

---

## Project Structure

```
HardwareDoc_Final/
├── app.py                              # Flask application — all routes
├── requirements.txt                    # Python dependencies
├── engine/
│   ├── backward_chaining.py            # Inference engine (backward chaining)
│   └── knowledge_base.py              # KB loader + validator
├── database/
│   └── history.py                     # SQLite history module
├── data/
│   ├── rules.json                     # 20 IF-THEN production rules
│   ├── questions.json                 # 23 questions + 5 diagnostic goals
│   └── hardware_diagnostic_dataset.json  # 20 educational case records
├── templates/
│   ├── base.html                      # Base layout with sidebar
│   ├── home.html                      # Home page
│   ├── diagnosis_new.html             # Goal selection
│   ├── diagnosis_question.html        # Question page (WHY facility)
│   ├── diagnosis_result.html          # Result page
│   ├── diagnosis_how.html             # HOW explanation page
│   ├── history.html                   # History list
│   ├── history_detail.html            # Session detail
│   ├── knowledge.html                 # Knowledge base & dataset
│   ├── about.html                     # About project
│   └── error.html                     # 404/500 error page
├── static/
│   ├── css/style.css                  # Complete CSS design system
│   └── js/app.js                      # Mobile sidebar + UX JS
└── tests/
    └── test_all.py                    # 52 automated tests
```

---

## Windows — Quick Start

### Step 1 — Create and activate virtual environment
```powershell
cd "C:\Users\shahzeb baig\OneDrive\Desktop\HardwareDoc_Final"
python -m venv venv
venv\Scripts\activate
```

### Step 2 — Install dependencies
```powershell
pip install -r requirements.txt
```

### Step 3 — Run the tests (optional but recommended)
```powershell
pip install pytest
python -m pytest tests\test_all.py -v
```
Expected: **52 passed**

### Step 4 — Start the application
```powershell
python app.py
```

### Step 5 — Open in browser
```
http://127.0.0.1:5000
```

> **Note:** The SQLite database (`database/history.db`) is created automatically on first run. No configuration is required.

---

## Diagnostic Categories

| Goal | Possible Conclusions |
|---|---|
| Computer Does Not Power On | Disconnected cable, faulty button, PSU failure, motherboard fault |
| Powers On But No Display | Monitor off, cable disconnected, GPU fault, RAM seating |
| RAM-Related Startup Failure | RAM unseated, module failure, compatibility issue |
| Overheating / Unexpected Shutdown | CPU fan failure, thermal paste, blocked airflow |
| Storage Drive or Boot Failure | Data cable, mechanical HDD failure, OS boot failure, electronic failure |

---

## How Backward Chaining Works Here

1. User selects a **diagnostic goal** (e.g., "Computer Does Not Power On").
2. The engine looks up all rules whose **conclusions** match the goal's target faults.
3. For each rule, it tries to prove every **premise**:
   - If a premise is already known (True/False), it uses it directly.
   - If a premise is provable by another rule, it recurses.
   - If unknown and askable, it **asks the user a question**.
4. If all premises of a rule are proved → the conclusion is **established**.
5. If no rule fires → the engine tries the next target conclusion.
6. If no conclusion can be established → returns **"Insufficient Evidence"**.

Cycle detection prevents infinite recursion. Unknown answers are never silently converted to No.

---

## WHY Explanation

When a question is shown, clicking **"Why is this being asked?"** fetches a live AJAX response from `/diagnosis/why`. The response includes:

- The current diagnostic goal being investigated
- Which rule requires this fact as a premise
- The human-readable reason the fact matters
- All facts already established in this session

The explanation is **generated dynamically** from the actual inference state — not pre-written text.

## HOW Explanation

After a diagnosis, clicking **"How was this concluded?"** shows `/diagnosis/how` with:

- A numbered proof trace (goal → rules evaluated → premises checked → conclusion)
- Each rule that **fired** (green) and each rule that **failed** (grey)
- The premises checked for each rule with their True/False/Unknown values
- Supporting facts marked with ★ if they contributed to a fired rule

Only rules that actually fired are shown as successful.

---

## Dataset vs. Knowledge Base vs. Working Memory

| Component | What it is | Where stored |
|---|---|---|
| **Knowledge Base** | IF-THEN rules authored by a domain expert | `data/rules.json` |
| **Dataset** | Educational case records illustrating expected reasoning paths | `data/hardware_diagnostic_dataset.json` |
| **Working Memory** | Facts known during the current session (session-scoped, not persisted) | Flask session (in-memory) |

The dataset is **validated at startup**: rule IDs, fact IDs, outcome labels, and duplicate case IDs are all checked.

---

## Test Results (52 tests, all passing)

| Test Class | Tests | Coverage |
|---|---|---|
| `TestBackwardChaining` | 11 | Successful proof, alternative rules, unknown ≠ no, go-back, cycles |
| `TestWHYExplanation` | 3 | Goal, rules, known facts in WHY |
| `TestHOWExplanation` | 5 | Goal, conclusion, steps, facts, incomplete session |
| `TestDatasetLoading` | 8 | Loading, unique IDs, rule refs, fact refs, invalid detection |
| `TestSQLiteHistory` | 10 | Save, retrieve, persist, delete, clear, empty state, snapshot |
| `TestFlaskRoutes` | 10 | All routes, invalid form inputs, 404 handler |
| `TestRealKBInferenceEngine` | 5 | 5 real KB diagnosis scenarios end-to-end |

---

## Safety Disclaimer

This application is for **educational use only**. All diagnoses are produced by a rule-based system with limited observable symptoms and are not a substitute for professional hardware testing. The application does not instruct users to open power supplies or touch live components.

---

## Known Limitations

- The knowledge base covers 5 hardware fault categories with 20 rules. Edge cases outside these categories return "Insufficient Evidence".
- The diagnostic flow requires the user to have physical access to the computer and some basic hardware knowledge (e.g., locating the motherboard standby LED).
- Flash/session state is stored in a server-side cookie; sessions are lost if the server restarts mid-diagnosis.
- The Playwright browser driver could not be installed in this environment, so visual UI verification was done via HTTP response codes rather than screenshots.
