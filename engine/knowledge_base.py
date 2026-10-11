"""
Knowledge Base Loader
=====================
Loads and validates all knowledge base data files at application startup.
Provides a single access point for rules, questions, goals, and the dataset.
"""

import json
import os
import logging

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")


def _load_json(filename: str) -> dict | list:
    """Load a JSON file from the data directory."""
    path = os.path.join(DATA_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_knowledge_base() -> dict:
    """
    Load and validate all knowledge base components.
    
    Returns a dict with keys:
      - rules: list of rule dicts
      - questions: list of question dicts
      - diagnostic_goals: list of goal dicts
      - fault_descriptions: dict mapping conclusion_id -> info
      - dataset: the full dataset dict
      - validation_errors: list of validation error strings
      - validation_warnings: list of validation warning strings
    """
    errors = []
    warnings = []

    # Load files
    try:
        rules_data = _load_json("rules.json")
        rules = rules_data["rules"]
        fault_descriptions = rules_data["fault_descriptions"]
    except Exception as e:
        rules = []
        fault_descriptions = {}
        errors.append(f"Failed to load rules.json: {e}")

    try:
        questions_data = _load_json("questions.json")
        questions = questions_data["questions"]
        diagnostic_goals = questions_data["diagnostic_goals"]
    except Exception as e:
        questions = []
        diagnostic_goals = []
        errors.append(f"Failed to load questions.json: {e}")

    try:
        dataset = _load_json("hardware_diagnostic_dataset.json")
    except Exception as e:
        dataset = {"cases": [], "metadata": {}}
        errors.append(f"Failed to load hardware_diagnostic_dataset.json: {e}")

    # Collect all known fact IDs from rules
    all_premise_facts = set()
    all_conclusion_ids = set()
    rule_ids = set()
    for rule in rules:
        if "id" not in rule:
            errors.append(f"Rule missing 'id' field: {rule}")
            continue
        if rule["id"] in rule_ids:
            errors.append(f"Duplicate rule ID: {rule['id']}")
        rule_ids.add(rule["id"])
        all_conclusion_ids.add(rule.get("conclusion", ""))
        for p in rule.get("premises", []):
            all_premise_facts.add(p)

    # Collect all fact IDs from questions
    question_fact_ids = set()
    question_ids = set()
    for q in questions:
        if "id" not in q:
            errors.append(f"Question missing 'id': {q}")
            continue
        if q["id"] in question_ids:
            errors.append(f"Duplicate question ID: {q['id']}")
        question_ids.add(q["id"])
        if "fact_id" in q:
            question_fact_ids.add(q["fact_id"])
        for f in q.get("answer_yes_sets", []) + q.get("answer_no_sets", []):
            question_fact_ids.add(f)

    # Check: All premise facts that aren't conclusions should have questions
    for fact in all_premise_facts:
        if fact not in all_conclusion_ids and fact not in question_fact_ids:
            warnings.append(
                f"Premise fact '{fact}' is not a rule conclusion and has no question defined."
            )

    # Validate fault_descriptions covers all conclusions
    for conclusion_id in all_conclusion_ids:
        if conclusion_id not in fault_descriptions:
            warnings.append(
                f"Conclusion '{conclusion_id}' has no entry in fault_descriptions."
            )

    # Validate dataset
    valid_statuses = {"confirmed", "suspected", "insufficient_evidence"}
    case_ids = set()
    dataset_errors = []
    dataset_warnings = []

    for case in dataset.get("cases", []):
        case_id = case.get("id", "UNKNOWN")

        if case_id in case_ids:
            dataset_errors.append(f"Dataset: Duplicate case ID '{case_id}'")
        case_ids.add(case_id)

        # Validate expected_status
        status = case.get("expected_status")
        if status and status not in valid_statuses:
            dataset_errors.append(
                f"Dataset case '{case_id}': invalid expected_status '{status}'"
            )

        # Validate rule IDs
        for rule_id in case.get("applicable_rules", []):
            if rule_id not in rule_ids:
                dataset_errors.append(
                    f"Dataset case '{case_id}': references unknown rule ID '{rule_id}'"
                )

        # Validate fact IDs
        for fact_id in case.get("symptom_facts", []):
            if fact_id not in all_premise_facts and fact_id not in all_conclusion_ids and fact_id not in question_fact_ids:
                dataset_warnings.append(
                    f"Dataset case '{case_id}': symptom fact '{fact_id}' not found in knowledge base"
                )

        # Validate expected_conclusion
        expected = case.get("expected_conclusion")
        if expected and expected not in all_conclusion_ids:
            dataset_errors.append(
                f"Dataset case '{case_id}': expected_conclusion '{expected}' not in rules"
            )

    errors.extend(dataset_errors)
    warnings.extend(dataset_warnings)

    if errors:
        logger.error("Knowledge base validation errors: %s", errors)
    if warnings:
        logger.warning("Knowledge base validation warnings: %s", warnings)

    return {
        "rules": rules,
        "questions": questions,
        "diagnostic_goals": diagnostic_goals,
        "fault_descriptions": fault_descriptions,
        "dataset": dataset,
        "validation_errors": errors,
        "validation_warnings": warnings,
        "rule_ids": rule_ids,
        "all_fact_ids": all_premise_facts | question_fact_ids,
    }
