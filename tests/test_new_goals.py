"""
Regression tests for the four new diagnostic goals:
- goal_gpu_display   (Graphics Card or Display Output Failure)
- goal_cpu_cooling   (CPU Cooling or Fan Failure)
- goal_ram_errors    (RAM Detection or Memory Errors)
- goal_bios_post     (BIOS / POST Startup Failure)

Tests verify:
1. Each goal starts correctly and enters the inference flow.
2. Known answer sequences reach the expected proved conclusion.
3. The new BIOS/POST rules (R21-R23) fire correctly.
4. The new goals start correctly via Flask /diagnosis/start.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.knowledge_base import load_knowledge_base
from engine.backward_chaining import (
    InferenceEngine, FACT_TRUE, FACT_FALSE, FACT_UNKNOWN
)
import app as flask_app


def make_engine():
    kb = load_knowledge_base()
    return InferenceEngine(
        rules=kb['rules'],
        questions=kb['questions'],
        diagnostic_goals=kb['diagnostic_goals'],
        fault_descriptions=kb['fault_descriptions'],
    )


class TestNewGoalInference(unittest.TestCase):
    """Inference engine tests for the four new goals."""

    def setUp(self):
        self.engine = make_engine()

    # ── goal_gpu_display ──────────────────────────────────────────────────────

    def test_gpu_display_goal_exists(self):
        """goal_gpu_display must be registered in the engine."""
        self.assertIn('goal_gpu_display', self.engine.diagnostic_goals)

    def test_gpu_display_initial_fact_set(self):
        """goal_gpu_display must set computer_powers_on as initial fact."""
        s = self.engine.create_session('goal_gpu_display')
        self.assertEqual(s['working_memory'].get('computer_powers_on'), FACT_TRUE)

    def test_gpu_display_monitor_off_proved(self):
        """Monitor off diagnosis via goal_gpu_display (R05)."""
        s = self.engine.create_session('goal_gpu_display')
        self.engine.process_answer(s, 'Q06', 'yes')  # no_display_on_monitor
        self.engine.process_answer(s, 'Q07', 'no')   # monitor_powered_off
        result = self.engine.get_next_question(s)
        self.assertEqual(result['status'], 'proved')
        self.assertEqual(result['proved_conclusion'], 'fault_monitor_off')

    def test_gpu_display_cable_fault_proved(self):
        """Display cable disconnected via goal_gpu_display (R06)."""
        s = self.engine.create_session('goal_gpu_display')
        self.engine.process_answer(s, 'Q06', 'yes')  # no_display_on_monitor
        self.engine.process_answer(s, 'Q07', 'no')   # monitor_powered_on (neg) NO wait:
        # Q07 fact_id: monitor_powered_on -> answer_yes means monitor IS on
        # We need monitor ON and cable disconnected for R06
        s2 = self.engine.create_session('goal_gpu_display')
        self.engine.process_answer(s2, 'Q06', 'yes')  # no_display_on_monitor
        self.engine.process_answer(s2, 'Q07', 'yes')  # monitor IS powered on
        self.engine.process_answer(s2, 'Q08', 'no')   # display_cable_disconnected (Q08 yes_sets: display_cable_connected)
        result = self.engine.get_next_question(s2)
        self.assertEqual(result['status'], 'proved')
        self.assertEqual(result['proved_conclusion'], 'fault_display_cable_disconnected')

    def test_gpu_display_gpu_suspected_proved(self):
        """GPU suspected via goal_gpu_display (R07): no display, monitor on, cable connected, beep codes."""
        s = self.engine.create_session('goal_gpu_display')
        self.engine.process_answer(s, 'Q06', 'yes')  # no_display_on_monitor
        self.engine.process_answer(s, 'Q07', 'yes')  # monitor powered on
        self.engine.process_answer(s, 'Q08', 'no')   # display_cable_connected (answer no = sets display_cable_disconnected)
        # Wait - Q08 fact_id is display_cable_connected
        # answer_yes_sets: display_cable_connected, answer_no_sets: display_cable_disconnected
        # R07 needs display_cable_connected - so answer Q08 = yes
        s2 = self.engine.create_session('goal_gpu_display')
        self.engine.process_answer(s2, 'Q06', 'yes')  # no_display_on_monitor
        self.engine.process_answer(s2, 'Q07', 'yes')  # monitor powered on
        self.engine.process_answer(s2, 'Q08', 'yes')  # display_cable_connected
        self.engine.process_answer(s2, 'Q09', 'yes')  # gpu_beep_codes
        result = self.engine.get_next_question(s2)
        self.assertEqual(result['status'], 'proved')
        self.assertEqual(result['proved_conclusion'], 'fault_gpu_suspected')

    # ── goal_cpu_cooling ──────────────────────────────────────────────────────

    def test_cpu_cooling_goal_exists(self):
        self.assertIn('goal_cpu_cooling', self.engine.diagnostic_goals)

    def test_cpu_cooling_fan_fault_confirmed(self):
        """CPU fan fault via goal_cpu_cooling (R13)."""
        s = self.engine.create_session('goal_cpu_cooling')
        self.engine.process_answer(s, 'Q14', 'yes')  # unexpected_shutdown
        self.engine.process_answer(s, 'Q15', 'yes')  # cpu_temperature_high
        self.engine.process_answer(s, 'Q16', 'no')   # cpu_fan_not_spinning
        result = self.engine.get_next_question(s)
        self.assertEqual(result['status'], 'proved')
        self.assertEqual(result['proved_conclusion'], 'fault_cpu_fan')

    def test_cpu_cooling_thermal_paste_suspected(self):
        """Thermal paste fault via goal_cpu_cooling (R14)."""
        s = self.engine.create_session('goal_cpu_cooling')
        self.engine.process_answer(s, 'Q14', 'yes')  # unexpected_shutdown
        self.engine.process_answer(s, 'Q15', 'yes')  # cpu_temperature_high
        self.engine.process_answer(s, 'Q16', 'yes')  # cpu_fan_spinning
        self.engine.process_answer(s, 'Q17', 'yes')  # thermal_paste_dried
        result = self.engine.get_next_question(s)
        self.assertEqual(result['status'], 'proved')
        self.assertEqual(result['proved_conclusion'], 'fault_thermal_paste')

    def test_cpu_cooling_poor_airflow_suspected(self):
        """Poor airflow via goal_cpu_cooling (R15)."""
        s = self.engine.create_session('goal_cpu_cooling')
        self.engine.process_answer(s, 'Q14', 'yes')  # unexpected_shutdown
        self.engine.process_answer(s, 'Q15', 'no')   # cpu_temperature NOT high -> R12/R13/R14 fail
        self.engine.process_answer(s, 'Q18', 'yes')  # case_vents_blocked
        result = self.engine.get_next_question(s)
        self.assertEqual(result['status'], 'proved')
        self.assertEqual(result['proved_conclusion'], 'fault_poor_airflow')

    # ── goal_ram_errors ──────────────────────────────────────────────────────

    def test_ram_errors_goal_exists(self):
        self.assertIn('goal_ram_errors', self.engine.diagnostic_goals)

    def test_ram_errors_unseated_confirmed(self):
        """RAM unseated via goal_ram_errors (R09)."""
        s = self.engine.create_session('goal_ram_errors')
        self.engine.process_answer(s, 'Q11', 'yes')  # beep_on_startup
        self.engine.process_answer(s, 'Q12', 'yes')  # ram_beep_code
        self.engine.process_answer(s, 'Q10', 'yes')  # ram_not_seated
        result = self.engine.get_next_question(s)
        self.assertEqual(result['status'], 'proved')
        self.assertEqual(result['proved_conclusion'], 'fault_ram_unseated')

    def test_ram_errors_module_suspected(self):
        """RAM module fault via goal_ram_errors (R10)."""
        s = self.engine.create_session('goal_ram_errors')
        self.engine.process_answer(s, 'Q11', 'yes')  # beep_on_startup
        self.engine.process_answer(s, 'Q12', 'yes')  # ram_beep_code
        self.engine.process_answer(s, 'Q10', 'no')   # ram IS seated -> fault_ram_module_suspected
        result = self.engine.get_next_question(s)
        self.assertEqual(result['status'], 'proved')
        self.assertEqual(result['proved_conclusion'], 'fault_ram_module_suspected')

    def test_ram_errors_compatibility_suspected(self):
        """RAM compatibility error via goal_ram_errors (R11)."""
        s = self.engine.create_session('goal_ram_errors')
        self.engine.process_answer(s, 'Q11', 'no')   # no_beep_on_startup
        self.engine.process_answer(s, 'Q13', 'yes')  # ram_error_on_screen
        result = self.engine.get_next_question(s)
        self.assertEqual(result['status'], 'proved')
        self.assertEqual(result['proved_conclusion'], 'fault_ram_compatibility')

    # ── goal_bios_post ────────────────────────────────────────────────────────

    def test_bios_post_goal_exists(self):
        self.assertIn('goal_bios_post', self.engine.diagnostic_goals)

    def test_bios_post_initial_fact_set(self):
        """goal_bios_post must set computer_powers_on as initial fact."""
        s = self.engine.create_session('goal_bios_post')
        self.assertEqual(s['working_memory'].get('computer_powers_on'), FACT_TRUE)

    def test_bios_post_failure_via_beep_codes(self):
        """POST failure via R21: beep codes + blank screen."""
        s = self.engine.create_session('goal_bios_post')
        self.engine.process_answer(s, 'Q24', 'yes')  # post_beep_code_heard
        self.engine.process_answer(s, 'Q26', 'yes')  # post_no_video_no_beep
        result = self.engine.get_next_question(s)
        self.assertEqual(result['status'], 'proved')
        self.assertEqual(result['proved_conclusion'], 'fault_post_failure')

    def test_bios_post_failure_via_silent(self):
        """POST failure via R22: no beep codes + blank screen."""
        s = self.engine.create_session('goal_bios_post')
        self.engine.process_answer(s, 'Q24', 'no')   # no_post_beep_code
        self.engine.process_answer(s, 'Q26', 'yes')  # post_no_video_no_beep
        result = self.engine.get_next_question(s)
        self.assertEqual(result['status'], 'proved')
        self.assertEqual(result['proved_conclusion'], 'fault_post_failure')

    def test_bios_post_cmos_battery_confirmed(self):
        """CMOS battery flat via R23: no beep, video works, date resets."""
        s = self.engine.create_session('goal_bios_post')
        self.engine.process_answer(s, 'Q24', 'no')   # no_post_beep_code
        self.engine.process_answer(s, 'Q26', 'no')   # video IS present (not blank)
        self.engine.process_answer(s, 'Q25', 'yes')  # bios_date_reset_on_boot
        result = self.engine.get_next_question(s)
        self.assertEqual(result['status'], 'proved')
        self.assertEqual(result['proved_conclusion'], 'fault_cmos_battery')

    def test_bios_post_rules_have_fault_descriptions(self):
        """fault_post_failure and fault_cmos_battery must have fault descriptions."""
        kb = load_knowledge_base()
        self.assertIn('fault_post_failure', kb['fault_descriptions'])
        self.assertIn('fault_cmos_battery', kb['fault_descriptions'])

    # ── Flask route: new goals start correctly ────────────────────────────────

    def test_new_goals_appear_in_flask_context(self):
        """All 4 new goals must appear in the Flask /diagnosis/new page."""
        flask_app.app.config['TESTING'] = True
        flask_app.app.config['SECRET_KEY'] = 'test-key'
        client = flask_app.app.test_client()
        resp = client.get('/diagnosis/new')
        self.assertEqual(resp.status_code, 200)
        body = resp.data.decode()
        self.assertIn('goal_gpu_display', body)
        self.assertIn('goal_cpu_cooling', body)
        self.assertIn('goal_ram_errors', body)
        self.assertIn('goal_bios_post', body)

    def test_goal_gpu_display_starts_via_flask(self):
        flask_app.app.config['TESTING'] = True
        flask_app.app.config['SECRET_KEY'] = 'test-key'
        client = flask_app.app.test_client()
        resp = client.post('/diagnosis/start',
                           data={'goal_id': 'goal_gpu_display'},
                           follow_redirects=True)
        self.assertEqual(resp.status_code, 200)
        # Should reach the question page, not flash an error
        body = resp.data.decode()
        self.assertNotIn('Invalid goal', body)
        self.assertIn('question_id', body)

    def test_goal_bios_post_starts_via_flask(self):
        flask_app.app.config['TESTING'] = True
        flask_app.app.config['SECRET_KEY'] = 'test-key'
        client = flask_app.app.test_client()
        resp = client.post('/diagnosis/start',
                           data={'goal_id': 'goal_bios_post'},
                           follow_redirects=True)
        self.assertEqual(resp.status_code, 200)
        body = resp.data.decode()
        self.assertNotIn('Invalid goal', body)
        self.assertIn('question_id', body)


if __name__ == '__main__':
    unittest.main(verbosity=2)
