
from unifusion.llm import answer, build_prompt, format_evidence, strip_reasoning


def test_evidence_is_numbered_and_truncated():
    rendered = format_evidence(["abcdefgh", "ij"], max_chars_per_passage=4)
    assert rendered == "[doc1] abcd\n\n[doc2] ij"


def test_empty_context_is_explicit():
    assert format_evidence([]) == "(no evidence retrieved)"


def test_prompt_carries_query_and_evidence():
    prompt = build_prompt("who won", ["the match report"])
    assert "who won" in prompt
    assert "[doc1] the match report" in prompt
    assert prompt.rstrip().endswith("Short Answer:")


def test_strip_reasoning_removes_closed_block():
    raw = "<think>step 1\nstep 2</think>  Paris"
    assert strip_reasoning(raw) == "Paris"


def test_strip_reasoning_returns_empty_for_truncated_scratchpad():
    raw = "<think>step 1\nstep 2 and then the model ran out of"
    assert strip_reasoning(raw) == ""


def test_strip_reasoning_handles_bare_prefix():
    assert strip_reasoning("Thinking process:\n\n42") == "42"


def test_strip_reasoning_is_idempotent_on_plain_answers():
    assert strip_reasoning("  Paris \n") == "Paris"
    assert strip_reasoning("Paris") == "Paris"


def test_strip_reasoning_truncates_long_answers():
    assert len(strip_reasoning("x" * 500, max_chars=50)) == 50


class _StubGenerator:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def generate(self, prompt, num_predict=None):
        self.calls.append(num_predict)
        return self.outputs.pop(0)


def test_answer_retries_once_when_empty():
    stub = _StubGenerator(["", "Paris"])
    assert answer(stub, "capital of France", ["[doc] Paris is the capital"], retry_num_predict=4096) == "Paris"
    assert stub.calls == [None, 4096]


def test_answer_does_not_retry_when_successful():
    stub = _StubGenerator(["Paris"])
    assert answer(stub, "q", ["ctx"], retry_num_predict=4096) == "Paris"
    assert stub.calls == [None]
