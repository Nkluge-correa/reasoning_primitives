"""
tests/test_parsers.py — unit tests for answer parsing and JSON extraction.

Tests the three parser functions used in the pipeline:
  - parse_reasoning_trace  (inference.py)
  - extract_json           (eval.py → score_sample)
  - normalize_answer       (eval.py → score_sample)

And the full pipeline as used in score_sample in eval.py:
  raw_output
      → parse_reasoning_trace()   [inference.py]
      → extract_json()            [eval.py]
      → normalize_answer()        [eval.py]
      → compare to correct_option

Run with:
    cd src
    python -m pytest tests/test_parsers.py -v

Run a single class:
    python -m pytest tests/test_parsers.py::TestNormalizeAnswer -v
    python -m pytest tests/test_parsers.py::TestScoreSamplePipeline -v
    python -m pytest tests/test_parsers.py::TestWithRealSamples -v
"""

import sys
import os
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from utils import normalize_answer, extract_json, parse_reasoning_trace
from templates import get_task


# ============================================================================
# normalize_answer
# ============================================================================

class TestNormalizeAnswer:

    # --- basic cases ---
    def test_plain_uppercase(self):
        for letter in ["A", "B", "C", "D"]:
            assert normalize_answer(letter) == letter

    def test_plain_lowercase(self):
        assert normalize_answer("a") == "A"
        assert normalize_answer("b") == "B"
        assert normalize_answer("c") == "C"
        assert normalize_answer("d") == "D"

    def test_with_leading_trailing_whitespace(self):
        assert normalize_answer("  A  ") == "A"
        assert normalize_answer("\tB\n") == "B"

    # --- letter + paren format ---
    def test_letter_paren_no_text(self):
        assert normalize_answer("A)") == "A"

    def test_letter_paren_with_text(self):
        assert normalize_answer("A) some option text") == "A"
        assert normalize_answer("B) 42") == "B"

    def test_letter_paren_multiline(self):
        # Reported bug — requires re.DOTALL fix
        assert normalize_answer("A) some reasoning\nhere") == "A"
        assert normalize_answer("B) line1\nline2\nline3") == "B"

    def test_letter_paren_with_whitespace_before_text(self):
        assert normalize_answer("A)   lots of spaces") == "A"

    # --- invalid inputs ---
    def test_none(self):
        assert normalize_answer(None) is None

    def test_empty_string(self):
        assert normalize_answer("") is None

    def test_invalid_letter(self):
        assert normalize_answer("E") is None
        assert normalize_answer("Z") is None
        assert normalize_answer("1") is None

    def test_whitespace_only(self):
        assert normalize_answer("   ") is None

    # --- task-specific option values ---
    def test_olmo_original_bit_values(self):
        # olmo_original options are 0, 1, 2, 3 — answer extracted from JSON as letter
        assert normalize_answer("A") == "A"
        assert normalize_answer("D") == "D"

    def test_dyck_three_choice(self):
        # dyck is 3-choice A/B/C only
        assert normalize_answer("A") == "A"
        assert normalize_answer("B") == "B"
        assert normalize_answer("C") == "C"
        # normalize_answer doesn't know task context so D still normalizes
        assert normalize_answer("D") == "D"


# ============================================================================
# extract_json
# ============================================================================

class TestExtractJson:

    # --- clean JSON ---
    def test_clean_json_answer_a(self):
        result, _, _ = extract_json('{"answer": "A"}')
        assert result["answer"] == "A"

    def test_clean_json_answer_b(self):
        result, _, _ = extract_json('{"answer": "B"}')
        assert result["answer"] == "B"

    def test_clean_json_all_letters(self):
        for letter in ["A", "B", "C", "D"]:
            result, _, _ = extract_json(f'{{"answer": "{letter}"}}')
            assert result is not None
            assert result["answer"] == letter

    # --- answer key variants ---
    def test_answer_key_uppercase(self):
        result, _, _ = extract_json('{"Answer": "C"}')
        assert result is not None

    def test_answer_key_all_caps(self):
        result, _, _ = extract_json('{"ANSWER": "D"}')
        assert result is not None

    # --- JSON embedded in text ---
    def test_json_with_prefix_text(self):
        result, _, _ = extract_json('The answer is {"answer": "B"} based on my reasoning.')
        assert result is not None
        assert result["answer"] == "B"

    def test_json_with_newlines_around(self):
        result, _, _ = extract_json('\n\n{"answer": "A"}\n\n')
        assert result is not None

    # --- after parse_reasoning_trace strips thinking ---
    def test_after_think_tags_stripped(self):
        completion = '{"answer": "C"}'
        result, _, _ = extract_json(completion)
        assert result["answer"] == "C"

    # --- task-specific answer formats ---
    def test_olmo_original_numeric_answer(self):
        # olmo_original answer is a letter A/B/C/D, option values are 0/1/2/3
        result, _, _ = extract_json('{"answer": "D"}')
        assert result["answer"] == "D"

    def test_dyck_three_choice(self):
        for letter in ["A", "B", "C"]:
            result, _, _ = extract_json(f'{{"answer": "{letter}"}}')
            assert result is not None

    def test_collisions_answer(self):
        result, _, _ = extract_json('{"answer": "A"}')
        assert result["answer"] == "A"

    def test_dag_arithmetic_answer(self):
        result, _, _ = extract_json('{"answer": "B"}')
        assert result["answer"] == "B"

    # --- failure cases ---
    def test_no_json(self):
        result, _, _ = extract_json("no json here at all")
        assert result is None

    def test_empty_string(self):
        result, _, _ = extract_json("")
        assert result is None

    def test_does_not_crash_on_garbage(self):
        result, _, _ = extract_json("!!!###$$$%%%")
        assert True  # just check it doesn't raise


# ============================================================================
# parse_reasoning_trace
# ============================================================================

class TestParseReasoningTrace:

    # --- instruct models (no think tags) ---
    def test_instruct_clean_json(self):
        raw = '{"answer": "A"}'
        completion, reasoning = parse_reasoning_trace(raw)
        assert completion == '{"answer": "A"}'
        assert reasoning == ""

    def test_instruct_json_with_whitespace(self):
        raw = '  {"answer": "B"}  '
        completion, reasoning = parse_reasoning_trace(raw)
        assert completion == '{"answer": "B"}'
        assert reasoning == ""

    # --- thinking models (full think tags) ---
    def test_think_tags_basic(self):
        raw = '<think>some reasoning</think>\n{"answer": "C"}'
        completion, reasoning = parse_reasoning_trace(raw)
        assert completion == '{"answer": "C"}'
        assert reasoning == "some reasoning"

    def test_think_tags_multiline_reasoning(self):
        raw = '<think>\nline1\nline2\nline3\n</think>\n{"answer": "D"}'
        completion, reasoning = parse_reasoning_trace(raw)
        assert completion == '{"answer": "D"}'
        assert "line1" in reasoning
        assert "line3" in reasoning

    def test_think_tags_long_reasoning(self):
        long = "step reasoning " * 500
        raw = f"<think>{long}</think>\n{{\"answer\": \"A\"}}"
        completion, reasoning = parse_reasoning_trace(raw)
        assert completion == '{"answer": "A"}'
        assert len(reasoning) > 100

    def test_think_tags_empty_reasoning(self):
        raw = "<think></think>\n{\"answer\": \"B\"}"
        completion, reasoning = parse_reasoning_trace(raw)
        assert completion == '{"answer": "B"}'
        assert reasoning == ""

    def test_think_tags_whitespace_stripping(self):
        raw = "<think>reasoning</think>   \n\n   {\"answer\": \"C\"}   "
        completion, reasoning = parse_reasoning_trace(raw)
        assert completion == '{"answer": "C"}'
        assert reasoning == "reasoning"


# ============================================================================
# TestScoreSamplePipeline
# Mirrors exactly what score_sample in eval.py does.
# This is the most important test class — it tests the parsers
# in the exact context they are used in production.
# ============================================================================

class TestScoreSamplePipeline:
    """
    Reproduces the exact pipeline in eval.py → score_sample:

        raw_output (stored by inference.py after parse_reasoning_trace)
            → extract_json()
            → normalize_answer()
            → compare to correct_option
    """

    def _score(self, raw_output, correct_option):
        """Reproduce score_sample logic exactly as in eval.py."""
        # inference.py stores completion after parse_reasoning_trace
        completion, reasoning = parse_reasoning_trace(raw_output)

        # eval.py score_sample calls extract_json on completion
        parsed, _, _ = extract_json(completion)
        if parsed is None:
            return {"parse_failed": True, "is_correct": None, "model_choice": None}

        raw_ans = (
            parsed.get("answer") or
            parsed.get("Answer") or
            parsed.get("ANSWER") or ""
        )
        model_choice = normalize_answer(raw_ans)

        return {
            "parse_failed": model_choice is None,
            "model_choice": model_choice,
            "is_correct": model_choice == correct_option if model_choice else None,
        }

    # --- instruct model correct ---
    def test_instruct_correct_a(self):
        result = self._score('{"answer": "A"}', "A")
        assert result["parse_failed"] == False
        assert result["is_correct"] == True

    def test_instruct_correct_b(self):
        result = self._score('{"answer": "B"}', "B")
        assert result["parse_failed"] == False
        assert result["is_correct"] == True

    def test_instruct_correct_all(self):
        for letter in ["A", "B", "C", "D"]:
            result = self._score(f'{{"answer": "{letter}"}}', letter)
            assert result["is_correct"] == True

    # --- instruct model wrong ---
    def test_instruct_wrong(self):
        result = self._score('{"answer": "A"}', "B")
        assert result["parse_failed"] == False
        assert result["is_correct"] == False

    # --- thinking model correct ---
    def test_thinking_correct(self):
        raw = "<think>long reasoning...</think>\n{\"answer\": \"C\"}"
        result = self._score(raw, "C")
        assert result["parse_failed"] == False
        assert result["is_correct"] == True

    def test_thinking_wrong(self):
        raw = "<think>long reasoning...</think>\n{\"answer\": \"A\"}"
        result = self._score(raw, "C")
        assert result["parse_failed"] == False
        assert result["is_correct"] == False

    def test_thinking_long_trace(self):
        trace = "Let me think step by step. " * 200
        raw = f"<think>{trace}</think>\n{{\"answer\": \"D\"}}"
        result = self._score(raw, "D")
        assert result["parse_failed"] == False
        assert result["is_correct"] == True

    # --- parse failures ---
    def test_parse_failure_no_json(self):
        result = self._score("I don't know the answer.", "A")
        assert result["parse_failed"] == True
        assert result["is_correct"] == None

    def test_parse_failure_truncated_mid_think(self):
        # Model ran out of tokens mid-generation inside think block
        result = self._score("<think>reasoning cut off mid", "A")
        assert result["parse_failed"] == True

    def test_parse_failure_empty(self):
        result = self._score("", "A")
        assert result["parse_failed"] == True

    # --- the reported bug ---
    def test_multiline_answer_value_bug(self):
        # Reported by colleague: model outputs answer value with newline
        # normalize_answer needs re.DOTALL to handle this
        raw = '{"answer": "A) some text\nmore text"}'
        result = self._score(raw, "A")
        assert result["parse_failed"] == False
        assert result["is_correct"] == True

    def test_multiline_answer_b(self):
        raw = '{"answer": "B) option text\nsecond line"}'
        result = self._score(raw, "B")
        assert result["is_correct"] == True

    # --- task-specific: olmo_original ---
    def test_olmo_original_correct(self):
        # olmo_original: options have values 0/1/2/3, answer is A/B/C/D
        raw = '{"answer": "D"}'
        result = self._score(raw, "D")
        assert result["is_correct"] == True

    def test_olmo_original_thinking(self):
        raw = "<think>Initial: a=1, b=0. After swap: a=0, b=1. bits[0]=0.</think>\n{\"answer\": \"B\"}"
        result = self._score(raw, "B")
        assert result["is_correct"] == True

    # --- task-specific: collisions ---
    def test_collisions_correct(self):
        raw = "<think>A collides with B, exchanging velocities. Final A=20.</think>\n{\"answer\": \"A\"}"
        result = self._score(raw, "A")
        assert result["is_correct"] == True

    def test_collisions_instruct(self):
        raw = '{"answer": "C"}'
        result = self._score(raw, "C")
        assert result["is_correct"] == True

    # --- task-specific: astro ---
    def test_astro_correct(self):
        raw = "<think>After swaps, a=3.524749 → HD-209458 b.</think>\n{\"answer\": \"B\"}"
        result = self._score(raw, "B")
        assert result["is_correct"] == True

    # --- task-specific: dag_arithmetic ---
    def test_dag_arithmetic_correct(self):
        raw = "<think>v1_0=10, v1_1=5, v2_1=3+10=13.</think>\n{\"answer\": \"B\"}"
        result = self._score(raw, "B")
        assert result["is_correct"] == True

    # --- task-specific: dyck (3-choice) ---
    def test_dyck_correct_a(self):
        result = self._score('{"answer": "A"}', "A")
        assert result["is_correct"] == True

    def test_dyck_correct_b(self):
        result = self._score('{"answer": "B"}', "B")
        assert result["is_correct"] == True

    def test_dyck_correct_c(self):
        result = self._score('{"answer": "C"}', "C")
        assert result["is_correct"] == True

    def test_dyck_model_outputs_d(self):
        # If model outputs D on a dyck question it should be wrong (D is not a valid option)
        result = self._score('{"answer": "D"}', "A")
        assert result["is_correct"] == False
        assert result["parse_failed"] == False  # D is parsed but wrong


# ============================================================================
# TestWithRealSamples
# Generates actual samples from each task and verifies the full
# parser pipeline handles them correctly.
# ============================================================================

class TestWithRealSamples:
    """
    Generates real samples from each task generator and verifies
    the parser pipeline can correctly handle the generated format.
    """

    def _run_parser_check(self, task_name, m, n, csv_path=None, n_samples=5):
        """
        Generate n_samples real samples and verify the parser pipeline
        correctly handles perfect model output for each.
        """
        task = get_task(task_name, csv_path=csv_path)
        rng = random.Random(42)

        for _ in range(n_samples):
            sample = task.generate_sample(m=m, n=n, rng=rng)
            correct = sample["correct_option"]

            # Simulate perfect instruct model output
            raw_instruct = f'{{"answer": "{correct}"}}'
            completion, _ = parse_reasoning_trace(raw_instruct)
            parsed, _, _ = extract_json(completion)
            assert parsed is not None, \
                f"extract_json failed for {task_name} instruct output"
            answer = normalize_answer(parsed.get("answer", ""))
            assert answer == correct, \
                f"{task_name}: expected {correct}, got {answer}"

            # Simulate perfect thinking model output
            raw_think = f"<think>reasoning for {task_name}</think>\n{{\"answer\": \"{correct}\"}}"
            completion, reasoning = parse_reasoning_trace(raw_think)
            assert reasoning == f"reasoning for {task_name}"
            parsed, _, _ = extract_json(completion)
            assert parsed is not None
            answer = normalize_answer(parsed.get("answer", ""))
            assert answer == correct

        return sample  # return last sample for further checks

    def test_olmo_original_small(self):
        self._run_parser_check("olmo_original", m=5, n=4)

    def test_olmo_original_medium(self):
        self._run_parser_check("olmo_original", m=16, n=16)

    def test_collisions_small(self):
        self._run_parser_check("collisions", m=4, n=4)

    def test_collisions_medium(self):
        self._run_parser_check("collisions", m=16, n=16)

    def test_dag_arithmetic_small(self):
        self._run_parser_check("dag_arithmetic", m=2, n=2)

    def test_dag_arithmetic_medium(self):
        self._run_parser_check("dag_arithmetic", m=4, n=4)

    def test_dyck_small(self):
        sample = self._run_parser_check("dyck", m=1, n=8)
        # dyck is 3-choice — option_D should be empty
        assert sample.get("option_D", "") == "", \
            "dyck option_D should be empty (3-choice task)"
        assert sample["correct_option"] in {"A", "B", "C"}, \
            f"dyck correct_option should be A/B/C, got {sample['correct_option']}"

    def test_dyck_medium(self):
        sample = self._run_parser_check("dyck", m=4, n=32)
        assert sample["correct_option"] in {"A", "B", "C"}
        assert sample.get("option_D", "") == ""

    def test_dyck_correct_token_always_closer(self):
        """The correct answer for dyck must always be a closing bracket."""
        task = get_task("dyck")
        rng = random.Random(42)
        closers = {")", "]", "}"}
        for _ in range(20):
            sample = task.generate_sample(m=2, n=16, rng=rng)
            correct_lbl = sample["correct_option"]
            correct_val = sample[f"option_{correct_lbl}"]
            assert correct_val in closers, \
                f"dyck correct token should be a closer, got '{correct_val}'"

    def test_astro_small(self):
        csv_path = os.path.join(
            os.path.dirname(__file__), "..", "assets", "exoplanets.csv"
        )
        if not os.path.exists(csv_path):
            pytest.skip("exoplanets.csv not found — skipping astro tests")
        self._run_parser_check("astro", m=4, n=4, csv_path=csv_path)

    def test_astro_medium(self):
        csv_path = os.path.join(
            os.path.dirname(__file__), "..", "assets", "exoplanets.csv"
        )
        if not os.path.exists(csv_path):
            pytest.skip("exoplanets.csv not found — skipping astro tests")
        self._run_parser_check("astro", m=8, n=8, csv_path=csv_path)

    def test_all_tasks_correct_option_valid_label(self):
        """correct_option must always be A/B/C/D (A/B/C for dyck)."""
        tasks_to_test = [
            ("olmo_original", 5, 4),
            ("collisions",    4, 4),
            ("dag_arithmetic", 2, 2),
            ("dyck",          1, 8),
        ]
        for task_name, m, n in tasks_to_test:
            task = get_task(task_name)
            rng = random.Random(99)
            valid = {"A", "B", "C"} if task_name == "dyck" else {"A", "B", "C", "D"}
            for _ in range(10):
                sample = task.generate_sample(m=m, n=n, rng=rng)
                assert sample["correct_option"] in valid, \
                    f"{task_name}: correct_option '{sample['correct_option']}' not in {valid}"

    def test_all_tasks_options_are_nonempty(self):
        """All option_A/B/C/D should be non-empty except dyck's option_D."""
        tasks_to_test = [
            ("olmo_original", 5, 4),
            ("collisions",    4, 4),
            ("dag_arithmetic", 2, 2),
        ]
        for task_name, m, n in tasks_to_test:
            task = get_task(task_name)
            rng = random.Random(42)
            sample = task.generate_sample(m=m, n=n, rng=rng)
            for lbl in ["A", "B", "C", "D"]:
                assert sample.get(f"option_{lbl}", "") != "", \
                    f"{task_name}: option_{lbl} is empty"

    def test_dyck_option_d_empty(self):
        """dyck option_D must always be empty string."""
        task = get_task("dyck")
        rng = random.Random(42)
        for _ in range(10):
            sample = task.generate_sample(m=2, n=16, rng=rng)
            assert sample.get("option_D", "") == "", \
                "dyck option_D should always be empty"