"""
HardwareDoc — Expert System for Computer Hardware Fault Diagnosis
=================================================================
Main Flask application entry point.

Routes:
  GET  /                          Home page
  GET  /diagnosis/new             Select diagnostic goal
  POST /diagnosis/start           Start a new session
  GET  /diagnosis/question        Show next question
  POST /diagnosis/answer          Submit an answer
  POST /diagnosis/back            Go back to previous question
  GET  /diagnosis/result          Show final result
  GET  /diagnosis/why/<q_id>      WHY explanation (AJAX)
  GET  /diagnosis/how             HOW explanation
  GET  /history                   List all saved sessions
  GET  /history/<session_id>      View session detail
  POST /history/<session_id>/delete Delete a session
  POST /history/clear             Clear all history
  GET  /knowledge                 Knowledge base & dataset page
  GET  /about                     About page
  GET  /api/dataset               Dataset JSON API
"""

import os
import json
import copy
import logging
from datetime import datetime

from flask import (
    Flask, render_template, request, redirect, url_for,
    session as flask_session, jsonify, flash, abort
)

from engine.knowledge_base import load_knowledge_base
from engine.backward_chaining import InferenceEngine
from database.history import (
    init_db, save_session, get_all_sessions,
    get_session_detail, delete_session, delete_all_sessions, count_sessions
)

# ---------------------------------------------------------------------------
# App Setup
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "hardwaredoc-dev-secret-2024")
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load Knowledge Base at Startup
# ---------------------------------------------------------------------------

KB = load_knowledge_base()

if KB["validation_errors"]:
    logger.error("Knowledge base errors at startup: %s", KB["validation_errors"])
else:
    logger.info("Knowledge base loaded successfully. %d rules, %d questions, %d goals.",
                len(KB["rules"]), len(KB["questions"]), len(KB["diagnostic_goals"]))

ENGINE = InferenceEngine(
    rules=KB["rules"],
    questions=KB["questions"],
    diagnostic_goals=KB["diagnostic_goals"],
    fault_descriptions=KB["fault_descriptions"],
)

# Initialise DB
init_db()

# ---------------------------------------------------------------------------
# Context Processors
# ---------------------------------------------------------------------------

@app.context_processor
def inject_globals():
    """Inject common template variables."""
    return {
        "app_name": "HardwareDoc",
        "current_year": datetime.now().year,
        "kb_errors": KB["validation_errors"],
        "history_count": count_sessions(),
    }

# ---------------------------------------------------------------------------
# Home
# ---------------------------------------------------------------------------

@app.route("/")
def home():
    goals = KB["diagnostic_goals"]
    dataset_cases = KB.get("dataset", [])
    stats = {
        "rules":         len(KB["rules"]),
        "questions":     len(KB["questions"]),
        "goals":         len(goals),
        "dataset_cases": len(dataset_cases),
    }
    return render_template("home.html", goals=goals, stats=stats)

# ---------------------------------------------------------------------------
# New Diagnosis — Goal Selection
# ---------------------------------------------------------------------------

@app.route("/diagnosis/new")
def diagnosis_new():
    goals = KB["diagnostic_goals"]
    return render_template("diagnosis_new.html", goals=goals)

@app.route("/diagnosis/start", methods=["POST"])
def diagnosis_start():
    goal_id = request.form.get("goal_id", "").strip()
    if not goal_id or goal_id not in ENGINE.diagnostic_goals:
        flash("Please select a valid diagnostic goal.", "error")
        return redirect(url_for("diagnosis_new"))

    # Create session state
    diag_session = ENGINE.create_session(goal_id)
    flask_session["diagnosis"] = diag_session
    flask_session.modified = True

    return redirect(url_for("diagnosis_question"))

# ---------------------------------------------------------------------------
# Question Step
# ---------------------------------------------------------------------------

@app.route("/diagnosis/question")
def diagnosis_question():
    diag_session = flask_session.get("diagnosis")
    if not diag_session:
        flash("No active diagnosis session. Please start a new diagnosis.", "warning")
        return redirect(url_for("diagnosis_new"))

    if diag_session.get("status") == "complete" or diag_session.get("status") == "no_evidence":
        return redirect(url_for("diagnosis_result"))

    next_q = ENGINE.get_next_question(diag_session)

    if next_q["status"] == "proved":
        # Finalize the session
        proved_conclusion = next_q["proved_conclusion"]
        trace = next_q.get("trace", [])
        rules_used = next_q.get("rules_used", [])

        # Re-run to get full trace for finalize
        full_trace = []
        ENGINE._try_prove_goal(
            goal=proved_conclusion,
            working_memory=diag_session["working_memory"],
            asked_facts=set(diag_session["asked_facts"]),
            goal_stack=set(),
            trace=full_trace,
        )
        ENGINE.finalize_session(diag_session, proved_conclusion, rules_used, full_trace)
        flask_session["diagnosis"] = diag_session
        flask_session.modified = True

        # Save to DB
        try:
            saved_id = save_session(diag_session, diag_session["result"])
            flask_session["last_saved_session_id"] = saved_id
        except Exception as e:
            logger.error("Failed to save session: %s", e)

        return redirect(url_for("diagnosis_result"))

    elif next_q["status"] == "no_evidence":
        ENGINE.finalize_no_evidence(diag_session)
        flask_session["diagnosis"] = diag_session
        flask_session.modified = True
        try:
            saved_id = save_session(diag_session, diag_session["result"])
            flask_session["last_saved_session_id"] = saved_id
        except Exception as e:
            logger.error("Failed to save session: %s", e)
        return redirect(url_for("diagnosis_result"))

    elif next_q["status"] == "needs_question":
        question = next_q["question"]
        goal_config = ENGINE.diagnostic_goals.get(diag_session["goal_id"], {})
        questions_asked = ENGINE.count_questions_asked(diag_session)
        answers_log = diag_session.get("answers_log", [])

        # Store context for WHY explanation
        flask_session["why_context"] = {
            "question_id": question["id"],
            "goal_being_investigated": next_q.get("goal_being_investigated", ""),
            "relevant_rule": next_q.get("relevant_rule"),
            "rule_chain": next_q.get("rule_chain", []),
        }
        flask_session.modified = True

        return render_template(
            "diagnosis_question.html",
            question=question,
            goal=goal_config,
            questions_asked=questions_asked,
            answers_log=answers_log,
            goal_being_investigated=next_q.get("goal_being_investigated", ""),
        )

    else:
        ENGINE.finalize_no_evidence(diag_session)
        flask_session["diagnosis"] = diag_session
        flask_session.modified = True
        return redirect(url_for("diagnosis_result"))

# ---------------------------------------------------------------------------
# Answer Submission
# ---------------------------------------------------------------------------

@app.route("/diagnosis/answer", methods=["POST"])
def diagnosis_answer():
    diag_session = flask_session.get("diagnosis")
    if not diag_session:
        return redirect(url_for("diagnosis_new"))

    question_id = request.form.get("question_id", "").strip()
    answer = request.form.get("answer", "").strip().lower()

    if answer not in ("yes", "no", "unknown"):
        flash("Please select Yes, No, or Unknown.", "error")
        return redirect(url_for("diagnosis_question"))

    if not question_id or question_id not in ENGINE.questions_by_id:
        flash("Invalid question. Please try again.", "error")
        return redirect(url_for("diagnosis_question"))

    ENGINE.process_answer(diag_session, question_id, answer)
    flask_session["diagnosis"] = diag_session
    flask_session.modified = True

    return redirect(url_for("diagnosis_question"))

# ---------------------------------------------------------------------------
# Go Back
# ---------------------------------------------------------------------------

@app.route("/diagnosis/back", methods=["POST"])
def diagnosis_back():
    diag_session = flask_session.get("diagnosis")
    if not diag_session:
        return redirect(url_for("diagnosis_new"))

    ENGINE.undo_last_answer(diag_session)
    diag_session["status"] = "in_progress"
    diag_session["result"] = None
    flask_session["diagnosis"] = diag_session
    flask_session.modified = True

    return redirect(url_for("diagnosis_question"))

# ---------------------------------------------------------------------------
# Result Page
# ---------------------------------------------------------------------------

@app.route("/diagnosis/result")
def diagnosis_result():
    diag_session = flask_session.get("diagnosis")
    if not diag_session or not diag_session.get("result"):
        flash("No diagnosis result available.", "warning")
        return redirect(url_for("diagnosis_new"))

    result = diag_session["result"]
    saved_id = flask_session.get("last_saved_session_id")

    return render_template(
        "diagnosis_result.html",
        result=result,
        session_id=saved_id,
        goal=ENGINE.diagnostic_goals.get(diag_session["goal_id"], {}),
    )

# ---------------------------------------------------------------------------
# WHY Explanation (AJAX endpoint)
# ---------------------------------------------------------------------------

@app.route("/diagnosis/why")
def diagnosis_why():
    why_context = flask_session.get("why_context")
    diag_session = flask_session.get("diagnosis")

    if not why_context or not diag_session:
        return jsonify({"error": "No active question context available."})

    explanation = ENGINE.generate_why_explanation(
        session=diag_session,
        question_id=why_context["question_id"],
        goal_being_investigated=why_context["goal_being_investigated"],
        relevant_rule=why_context.get("relevant_rule"),
        rule_chain=why_context.get("rule_chain", []),
    )
    return jsonify(explanation)

# ---------------------------------------------------------------------------
# HOW Explanation
# ---------------------------------------------------------------------------

@app.route("/diagnosis/how")
def diagnosis_how():
    diag_session = flask_session.get("diagnosis")
    if not diag_session or not diag_session.get("result"):
        flash("No completed diagnosis available to explain.", "warning")
        return redirect(url_for("diagnosis_result"))

    how_explanation = ENGINE.generate_how_explanation(diag_session)

    return render_template(
        "diagnosis_how.html",
        explanation=how_explanation,
        result=diag_session["result"],
    )

# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

@app.route("/history")
def history():
    sessions = get_all_sessions()
    return render_template("history.html", sessions=sessions)

@app.route("/history/<session_id>")
def history_detail(session_id):
    record = get_session_detail(session_id)
    if not record:
        abort(404)
    return render_template("history_detail.html", record=record)

@app.route("/history/<session_id>/delete", methods=["POST"])
def history_delete(session_id):
    deleted = delete_session(session_id)
    if deleted:
        flash("Diagnosis record deleted successfully.", "success")
    else:
        flash("Record not found.", "error")
    return redirect(url_for("history"))

@app.route("/history/clear", methods=["POST"])
def history_clear():
    confirm = request.form.get("confirm", "").strip()
    if confirm != "yes":
        flash("Confirmation required to clear all history.", "error")
        return redirect(url_for("history"))
    count = delete_all_sessions()
    flash(f"All {count} diagnosis records cleared.", "success")
    return redirect(url_for("history"))

# ---------------------------------------------------------------------------
# Knowledge Base & Dataset
# ---------------------------------------------------------------------------

@app.route("/knowledge")
def knowledge():
    search = request.args.get("search", "").strip().lower()
    category_filter = request.args.get("category", "").strip().lower()

    cases = KB["dataset"].get("cases", [])

    # Filter by search
    if search:
        cases = [
            c for c in cases
            if search in c.get("description", "").lower()
            or search in c.get("id", "").lower()
            or search in c.get("category", "").lower()
            or search in c.get("expected_conclusion", "").lower()
        ]

    # Filter by category
    if category_filter:
        cases = [c for c in cases if c.get("category", "") == category_filter]

    categories = sorted(set(c.get("category", "") for c in KB["dataset"].get("cases", [])))

    return render_template(
        "knowledge.html",
        rules=KB["rules"],
        questions=KB["questions"],
        goals=KB["diagnostic_goals"],
        fault_descriptions=KB["fault_descriptions"],
        dataset_cases=cases,
        dataset_metadata=KB["dataset"].get("metadata", {}),
        validation_errors=KB["validation_errors"],
        validation_warnings=KB["validation_warnings"],
        search=search,
        category_filter=category_filter,
        categories=categories,
        total_cases=len(KB["dataset"].get("cases", [])),
    )

@app.route("/api/dataset")
def api_dataset():
    """Return the full dataset as JSON (for reference and testing)."""
    return jsonify(KB["dataset"])

# ---------------------------------------------------------------------------
# About
# ---------------------------------------------------------------------------

@app.route("/about")
def about():
    return render_template("about.html", rules=KB["rules"], goals=KB["diagnostic_goals"])

# ---------------------------------------------------------------------------
# Error Handlers
# ---------------------------------------------------------------------------

@app.errorhandler(404)
def not_found(e):
    return render_template("error.html", code=404, message="Page not found."), 404

@app.errorhandler(500)
def server_error(e):
    return render_template("error.html", code=500, message="Internal server error."), 500

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
