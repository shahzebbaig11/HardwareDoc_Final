"""
Backward Chaining Inference Engine
===================================
Implements a genuine backward-chaining expert system for hardware fault diagnosis.

Concepts:
- Goal: A fact (conclusion) we are trying to prove.
- Rule: IF premises THEN conclusion.
- Working Memory: Dictionary of known facts (True/False/Unknown).
- Goal Stack: Tracks what goals we are currently trying to prove.
- Inference Trace: Records every step taken during reasoning.

Algorithm:
1. Start with a diagnostic goal (e.g., prove fault_psu_suspected).
2. Find all rules whose conclusion matches the goal.
3. For each such rule, try to prove all premises.
4. If a premise is already known, use it directly.
5. If a premise is unknown, check if it can be proved by other rules.
6. If no rule can prove it, it must be asked from the user.
7. If a premise is disproved, backtrack and try the next rule.
8. If all premises of a rule are proved, the conclusion is proved.
9. If no rule can establish the conclusion, it is disproved.
"""

import json
import os
import copy

# Fact status constants
FACT_TRUE = "true"
FACT_FALSE = "false"
FACT_UNKNOWN = "unknown"

# Goal status constants
GOAL_PROVED = "proved"
GOAL_DISPROVED = "disproved"
GOAL_UNKNOWN = "unknown"
GOAL_NEEDS_INPUT = "needs_input"


class InferenceEngine:
    """
    Backward-chaining inference engine.
    
    This class is stateless across sessions — each diagnosis creates a new
    session state dictionary rather than modifying instance variables.
    This makes it safe for concurrent Flask requests.
    """

    def __init__(self, rules: list, questions: list, diagnostic_goals: list,
                 fault_descriptions: dict):
        """
        Initialize the engine with the knowledge base.
        
        Args:
            rules: List of rule dicts with id, premises, conclusion, confidence.
            questions: List of question dicts with id, fact_id, text, etc.
            diagnostic_goals: List of goal configs with target_conclusions.
            fault_descriptions: Dict mapping fault IDs to label/description/next_steps.
        """
        self.rules = rules
        self.fault_descriptions = fault_descriptions
        self.diagnostic_goals = {g["id"]: g for g in diagnostic_goals}

        # Index rules by their conclusion for fast lookup
        self.rules_by_conclusion: dict[str, list] = {}
        for rule in rules:
            conclusion = rule["conclusion"]
            if conclusion not in self.rules_by_conclusion:
                self.rules_by_conclusion[conclusion] = []
            self.rules_by_conclusion[conclusion].append(rule)

        # Index questions by their fact_id for quick lookup
        self.questions_by_fact: dict[str, dict] = {}
        self.questions_by_id: dict[str, dict] = {}
        for q in questions:
            self.questions_by_fact[q["fact_id"]] = q
            self.questions_by_id[q["id"]] = q

    # -------------------------------------------------------------------------
    # Session State Management
    # -------------------------------------------------------------------------

    def create_session(self, goal_id: str) -> dict:
        """
        Create a new diagnosis session state.
        
        Returns a serializable dictionary that represents the entire session.
        This dict is stored server-side (Flask session) and updated as the
        user answers questions.
        """
        goal_config = self.diagnostic_goals.get(goal_id)
        if not goal_config is None:
            initial_facts = {f: FACT_TRUE for f in goal_config.get("initial_facts", [])}
        else:
            initial_facts = {}

        return {
            "goal_id": goal_id,
            "working_memory": initial_facts,       # fact_id -> FACT_TRUE/FALSE/UNKNOWN
            "answers_log": [],                      # list of {question_id, fact_id, answer}
            "inference_trace": [],                  # list of trace step dicts
            "asked_facts": [],                      # list of fact_ids already asked
            "status": "in_progress",               # in_progress | complete | no_evidence
            "result": None,                        # Final result dict when complete
        }

    # -------------------------------------------------------------------------
    # Core Backward-Chaining Logic
    # -------------------------------------------------------------------------

    def get_next_question(self, session: dict) -> dict | None:
        """
        Determine the next question to ask the user.
        
        Runs backward chaining from all target conclusions of the goal.
        Returns the first unknown fact that is needed to continue inference,
        or None if inference is complete (either proved or cannot continue).
        
        Returns dict with keys:
          - question: the question object
          - goal_being_investigated: the conclusion we are trying to prove
          - relevant_rule: the rule that needs this fact
          - status: "needs_question" | "proved" | "disproved" | "no_evidence"
          - proved_conclusion: the conclusion ID if proved
        """
        goal_config = self.diagnostic_goals.get(session["goal_id"])
        if not goal_config:
            return {"status": "no_evidence"}

        target_conclusions = goal_config["target_conclusions"]
        working_memory = session["working_memory"]
        asked_facts = set(session["asked_facts"])

        # Try each target conclusion in order
        for conclusion_id in target_conclusions:
            result = self._try_prove_goal(
                goal=conclusion_id,
                working_memory=working_memory,
                asked_facts=asked_facts,
                goal_stack=set(),
                trace=[],
            )
            if result["status"] == GOAL_PROVED:
                return {
                    "status": "proved",
                    "proved_conclusion": conclusion_id,
                    "trace": result["trace"],
                    "rules_used": result["rules_used"],
                }
            elif result["status"] == GOAL_NEEDS_INPUT:
                # Return the question needed
                fact_id = result["needed_fact"]
                question = self.questions_by_fact.get(fact_id)
                if question:
                    return {
                        "status": "needs_question",
                        "question": question,
                        "goal_being_investigated": conclusion_id,
                        "relevant_rule": result.get("relevant_rule"),
                        "rule_chain": result.get("rule_chain", []),
                    }

        # No conclusion could be proved or needs more input
        # Check if any conclusion was already proved in working memory
        for conclusion_id in target_conclusions:
            if working_memory.get(conclusion_id) == FACT_TRUE:
                return {
                    "status": "proved",
                    "proved_conclusion": conclusion_id,
                    "trace": session.get("inference_trace", []),
                    "rules_used": [],
                }

        return {"status": "no_evidence"}

    def _try_prove_goal(self, goal: str, working_memory: dict, asked_facts: set,
                         goal_stack: set, trace: list) -> dict:
        """
        Attempt to prove a single goal using backward chaining.
        
        Returns a dict with:
          - status: GOAL_PROVED | GOAL_DISPROVED | GOAL_UNKNOWN | GOAL_NEEDS_INPUT
          - needed_fact: (if NEEDS_INPUT) the fact_id that must be asked
          - relevant_rule: (if NEEDS_INPUT) the rule being evaluated
          - rule_chain: list of rule IDs in the current chain
          - trace: list of reasoning steps
          - rules_used: list of rule IDs that contributed to the proof
        """
        # Cycle detection
        if goal in goal_stack:
            return {"status": GOAL_DISPROVED, "trace": trace, "rules_used": []}

        # Check working memory first
        if working_memory.get(goal) == FACT_TRUE:
            return {"status": GOAL_PROVED, "trace": trace, "rules_used": []}
        if working_memory.get(goal) == FACT_FALSE:
            return {"status": GOAL_DISPROVED, "trace": trace, "rules_used": []}

        # Check if this fact can be asked directly (it's a primitive fact, not a conclusion)
        if goal not in self.rules_by_conclusion:
            if goal in self.questions_by_fact:
                if goal not in asked_facts:
                    return {
                        "status": GOAL_NEEDS_INPUT,
                        "needed_fact": goal,
                        "relevant_rule": None,
                        "rule_chain": [],
                        "trace": trace,
                        "rules_used": [],
                    }
            return {"status": GOAL_UNKNOWN, "trace": trace, "rules_used": []}

        # Try each rule that could prove this goal
        new_goal_stack = goal_stack | {goal}
        rules_for_goal = self.rules_by_conclusion.get(goal, [])

        for rule in rules_for_goal:
            rule_trace = {
                "rule_id": rule["id"],
                "conclusion": goal,
                "premises_checked": [],
                "outcome": None,
            }
            premises_proved = True
            rules_used = []

            for premise in rule["premises"]:
                rule_trace["premises_checked"].append(premise)
                premise_result = self._try_prove_goal(
                    goal=premise,
                    working_memory=working_memory,
                    asked_facts=asked_facts,
                    goal_stack=new_goal_stack,
                    trace=trace,
                )

                if premise_result["status"] == GOAL_PROVED:
                    rules_used.extend(premise_result.get("rules_used", []))
                    continue
                elif premise_result["status"] == GOAL_NEEDS_INPUT:
                    # Propagate the need for input up
                    rule_trace["outcome"] = "needs_input"
                    trace.append(rule_trace)
                    result = premise_result.copy()
                    result["relevant_rule"] = rule
                    result["rule_chain"] = [rule["id"]] + premise_result.get("rule_chain", [])
                    return result
                elif premise_result["status"] in (GOAL_DISPROVED, GOAL_UNKNOWN):
                    premises_proved = False
                    break

            if premises_proved:
                rule_trace["outcome"] = "fired"
                trace.append(rule_trace)
                rules_used.append(rule["id"])
                return {
                    "status": GOAL_PROVED,
                    "trace": trace,
                    "rules_used": rules_used,
                }
            else:
                rule_trace["outcome"] = "failed"
                trace.append(rule_trace)

        return {"status": GOAL_DISPROVED, "trace": trace, "rules_used": []}

    # -------------------------------------------------------------------------
    # Answer Processing
    # -------------------------------------------------------------------------

    def process_answer(self, session: dict, question_id: str, answer: str) -> dict:
        """
        Process a user's answer to a question and update working memory.
        
        Args:
            session: Current session state (modified in place).
            question_id: ID of the question being answered.
            answer: "yes" | "no" | "unknown"
            
        Returns:
            Updated session dict.
        """
        question = self.questions_by_id.get(question_id)
        if not question:
            return session

        fact_id = question["fact_id"]

        # Record the answer
        answer_record = {
            "question_id": question_id,
            "fact_id": fact_id,
            "question_text": question["text"],
            "answer": answer,
        }

        # Remove any previous answer for this fact (for "go back" support)
        session["answers_log"] = [
            a for a in session["answers_log"] if a["fact_id"] != fact_id
        ]
        if fact_id in session["asked_facts"]:
            session["asked_facts"].remove(fact_id)

        session["answers_log"].append(answer_record)

        # Update working memory based on answer
        yes_sets = question.get("answer_yes_sets", [])
        no_sets  = question.get("answer_no_sets",  [])

        if answer == "yes":
            working_memory = session["working_memory"]
            # Only set fact_id=TRUE if it is not already handled by yes_sets
            if fact_id not in yes_sets:
                working_memory[fact_id] = FACT_TRUE
            for f in yes_sets:
                working_memory[f] = FACT_TRUE
            # Clear contradicting facts
            for f in no_sets:
                if f != fact_id:
                    working_memory.pop(f, None)
        elif answer == "no":
            working_memory = session["working_memory"]
            # Only set fact_id=FALSE if it is not already handled by no_sets
            if fact_id not in no_sets:
                working_memory[fact_id] = FACT_FALSE
            for f in no_sets:
                working_memory[f] = FACT_TRUE
            for f in yes_sets:
                if f != fact_id:
                    working_memory.pop(f, None)
        else:  # unknown
            session["working_memory"][fact_id] = FACT_UNKNOWN

        # Track that this fact has been asked
        if fact_id not in session["asked_facts"]:
            session["asked_facts"].append(fact_id)

        return session

    def undo_last_answer(self, session: dict) -> dict:
        """
        Remove the last answer and restore previous state.
        Allows the user to navigate backward.
        """
        if not session["answers_log"]:
            return session

        last_answer = session["answers_log"].pop()
        fact_id = last_answer["fact_id"]
        question = self.questions_by_id.get(last_answer["question_id"])

        # Remove from working memory
        session["working_memory"].pop(fact_id, None)
        if question:
            for f in question.get("answer_yes_sets", []):
                session["working_memory"].pop(f, None)
            for f in question.get("answer_no_sets", []):
                session["working_memory"].pop(f, None)

        # Remove from asked_facts
        if fact_id in session["asked_facts"]:
            session["asked_facts"].remove(fact_id)

        return session

    # -------------------------------------------------------------------------
    # Result Generation
    # -------------------------------------------------------------------------

    def finalize_session(self, session: dict, proved_conclusion: str,
                          rules_used: list, trace: list) -> dict:
        """
        Generate the final result for a completed diagnosis session.
        
        Returns a structured result dict suitable for display and storage.
        """
        fault_info = self.fault_descriptions.get(proved_conclusion, {})
        goal_config = self.diagnostic_goals.get(session["goal_id"], {})

        # Determine confidence from the rule that fired
        confidence = "suspected"
        fired_rules = []
        for rule in self.rules:
            if rule["id"] in rules_used:
                fired_rules.append(rule)
                if rule.get("confidence") == "confirmed":
                    confidence = "confirmed"
                elif rule.get("confidence") == "insufficient_evidence":
                    confidence = "insufficient_evidence"

        if confidence == "insufficient_evidence":
            status_label = "Insufficient Evidence"
            status_description = (
                "The available facts are not sufficient to identify a specific fault. "
                "Professional hardware testing is recommended."
            )
        elif confidence == "confirmed":
            status_label = "Confirmed by Rules"
            status_description = (
                "The diagnosis is directly supported by observable evidence "
                "and the knowledge base rules."
            )
        else:
            status_label = "Suspected"
            status_description = (
                "The diagnosis is strongly suggested by available evidence, "
                "but professional testing is required to confirm."
            )

        # Build known facts summary
        known_facts = []
        for ans in session["answers_log"]:
            known_facts.append({
                "fact_id": ans["fact_id"],
                "question": ans["question_text"],
                "answer": ans["answer"],
            })

        result = {
            "conclusion_id": proved_conclusion,
            "conclusion_label": fault_info.get("label", proved_conclusion),
            "conclusion_description": fault_info.get("description", ""),
            "confidence": confidence,
            "status_label": status_label,
            "status_description": status_description,
            "goal_id": session["goal_id"],
            "goal_label": goal_config.get("label", session["goal_id"]),
            "rules_used": rules_used,
            "fired_rules": [
                {"id": r["id"], "description": r["description"]} for r in fired_rules
            ],
            "known_facts": known_facts,
            "next_steps": fault_info.get("next_steps", []),
            "safety_note": fault_info.get("safety_note", ""),
            "inference_trace": trace,
        }

        session["status"] = "complete"
        session["result"] = result
        return session

    def finalize_no_evidence(self, session: dict) -> dict:
        """
        Finalize a session where no conclusion could be established.
        """
        goal_config = self.diagnostic_goals.get(session["goal_id"], {})

        known_facts = []
        for ans in session["answers_log"]:
            known_facts.append({
                "fact_id": ans["fact_id"],
                "question": ans["question_text"],
                "answer": ans["answer"],
            })

        result = {
            "conclusion_id": None,
            "conclusion_label": "Unable to Determine",
            "conclusion_description": (
                "The available information is not sufficient to identify "
                "a specific hardware fault using the current knowledge base."
            ),
            "confidence": "insufficient_evidence",
            "status_label": "Insufficient Evidence",
            "status_description": (
                "No rule in the knowledge base could be fully satisfied by the "
                "provided answers. Professional hardware diagnosis is recommended."
            ),
            "goal_id": session["goal_id"],
            "goal_label": goal_config.get("label", session["goal_id"]),
            "rules_used": [],
            "fired_rules": [],
            "known_facts": known_facts,
            "next_steps": [
                "Consult a qualified computer hardware technician.",
                "Request a comprehensive hardware test covering PSU, motherboard, and components.",
            ],
            "safety_note": (
                "Do not attempt internal repairs without proper training and tools."
            ),
            "inference_trace": session.get("inference_trace", []),
        }

        session["status"] = "no_evidence"
        session["result"] = result
        return session

    # -------------------------------------------------------------------------
    # WHY Explanation
    # -------------------------------------------------------------------------

    def generate_why_explanation(self, session: dict, question_id: str,
                                  goal_being_investigated: str,
                                  relevant_rule: dict | None,
                                  rule_chain: list) -> dict:
        """
        Generate a WHY explanation for a specific question.
        
        Explains why this question is being asked in the context of the
        current backward-chaining state.
        """
        question = self.questions_by_id.get(question_id)
        if not question:
            return {"error": "Question not found"}

        goal_config = self.diagnostic_goals.get(session["goal_id"], {})
        fault_info = self.fault_descriptions.get(goal_being_investigated, {})

        # Build rule chain descriptions
        rule_chain_descriptions = []
        for rule_id in rule_chain:
            for rule in self.rules:
                if rule["id"] == rule_id:
                    rule_chain_descriptions.append({
                        "id": rule["id"],
                        "description": rule["description"],
                        "premises": rule["premises"],
                        "conclusion": rule["conclusion"],
                    })

        # Find all rules that need this fact as a premise
        rules_needing_this_fact = []
        for rule in self.rules:
            if question["fact_id"] in rule["premises"]:
                # Check if this rule's conclusion is relevant to our goal
                related_to_goal = (
                    rule["conclusion"] == goal_being_investigated or
                    rule["conclusion"] in [
                        g for g in self.diagnostic_goals.get(
                            session["goal_id"], {}
                        ).get("target_conclusions", [])
                    ]
                )
                if related_to_goal:
                    rules_needing_this_fact.append({
                        "id": rule["id"],
                        "description": rule["description"],
                        "conclusion": rule["conclusion"],
                        "conclusion_label": self.fault_descriptions.get(
                            rule["conclusion"], {}
                        ).get("label", rule["conclusion"]),
                    })

        return {
            "question_id": question_id,
            "question_text": question["text"],
            "fact_id": question["fact_id"],
            "diagnostic_goal": {
                "id": session["goal_id"],
                "label": goal_config.get("label", session["goal_id"]),
            },
            "investigating_conclusion": {
                "id": goal_being_investigated,
                "label": fault_info.get("label", goal_being_investigated),
                "description": fault_info.get("description", ""),
            },
            "why_needed": question.get("why_explanation", ""),
            "rules_needing_this_fact": rules_needing_this_fact,
            "rule_chain": rule_chain_descriptions,
            "known_facts_so_far": [
                {
                    "fact_id": ans["fact_id"],
                    "answer": ans["answer"],
                    "question": ans["question_text"],
                }
                for ans in session["answers_log"]
            ],
        }

    # -------------------------------------------------------------------------
    # HOW Explanation
    # -------------------------------------------------------------------------

    def generate_how_explanation(self, session: dict) -> dict:
        """
        Generate a HOW explanation from the completed session.
        
        Produces a step-by-step account of how the inference engine
        reached its conclusion, referencing actual rules and facts.
        """
        result = session.get("result")
        if not result:
            return {"error": "No result available. Diagnosis is not yet complete."}

        goal_config = self.diagnostic_goals.get(session["goal_id"], {})
        trace = result.get("inference_trace", [])

        # Build proof steps from trace
        proof_steps = []
        step_num = 1

        # Step 1: Goal statement
        proof_steps.append({
            "step": step_num,
            "type": "goal",
            "description": (
                f"Diagnostic Goal: Investigate '{goal_config.get('label', session['goal_id'])}'"
            ),
        })
        step_num += 1

        # Steps for each rule evaluated
        for trace_step in trace:
            rule_id = trace_step.get("rule_id")
            outcome = trace_step.get("outcome")
            conclusion = trace_step.get("conclusion")
            premises = trace_step.get("premises_checked", [])

            rule_obj = None
            for r in self.rules:
                if r["id"] == rule_id:
                    rule_obj = r
                    break

            if outcome == "fired":
                premise_descriptions = []
                for p in premises:
                    fact_answer = session["working_memory"].get(p, FACT_UNKNOWN)
                    # Find the question text for this fact
                    q = self.questions_by_fact.get(p)
                    q_text = q["text"] if q else p
                    premise_descriptions.append({
                        "fact_id": p,
                        "text": q_text,
                        "value": fact_answer,
                    })

                proof_steps.append({
                    "step": step_num,
                    "type": "rule_fired",
                    "rule_id": rule_id,
                    "rule_description": rule_obj["description"] if rule_obj else rule_id,
                    "premises": premise_descriptions,
                    "conclusion": conclusion,
                    "conclusion_label": self.fault_descriptions.get(
                        conclusion, {}
                    ).get("label", conclusion),
                    "description": (
                        f"Rule {rule_id} fired: All premises were satisfied. "
                        f"Conclusion established: {self.fault_descriptions.get(conclusion, {}).get('label', conclusion)}"
                    ),
                })
                step_num += 1
            elif outcome == "failed":
                proof_steps.append({
                    "step": step_num,
                    "type": "rule_failed",
                    "rule_id": rule_id,
                    "rule_description": rule_obj["description"] if rule_obj else rule_id,
                    "description": (
                        f"Rule {rule_id} could not fire — one or more premises were not satisfied."
                    ),
                })
                step_num += 1

        # Final conclusion step
        proof_steps.append({
            "step": step_num,
            "type": "conclusion",
            "description": (
                f"Final Result: {result['status_label']} — "
                f"{result['conclusion_label']}"
            ),
            "conclusion_id": result["conclusion_id"],
            "conclusion_label": result["conclusion_label"],
            "status_label": result["status_label"],
            "confidence": result["confidence"],
        })

        # Collect all facts used
        supporting_facts = []
        for ans in session["answers_log"]:
            supporting_facts.append({
                "fact_id": ans["fact_id"],
                "question": ans["question_text"],
                "answer": ans["answer"],
                "is_supporting": ans["fact_id"] in [
                    p for r in self.rules
                    if r["id"] in result.get("rules_used", [])
                    for p in r["premises"]
                ],
            })

        return {
            "goal": {
                "id": session["goal_id"],
                "label": goal_config.get("label", session["goal_id"]),
                "description": goal_config.get("description", ""),
            },
            "conclusion": {
                "id": result["conclusion_id"],
                "label": result["conclusion_label"],
                "description": result["conclusion_description"],
                "status_label": result["status_label"],
                "status_description": result["status_description"],
                "confidence": result["confidence"],
            },
            "rules_used": result.get("fired_rules", []),
            "supporting_facts": supporting_facts,
            "proof_steps": proof_steps,
            "next_steps": result.get("next_steps", []),
            "safety_note": result.get("safety_note", ""),
        }

    def count_questions_asked(self, session: dict) -> int:
        """Return the number of questions asked so far."""
        return len(session["answers_log"])
