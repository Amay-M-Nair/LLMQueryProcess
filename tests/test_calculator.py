"""The calculator must be right, and must refuse anything it is not sure of.

A false positive here is worse than a false negative: it takes a real
question away from the model and answers it with a number.
"""

import pytest

from backend import calculator
from backend.calculator import NotArithmetic


@pytest.mark.parametrize(
    "question, expected",
    [
        ("what is 25% of 800?", 200),
        ("25 percent of 800", 200),
        ("2+2", 4),
        ("What is 17 * 23?", 391),
        ("calculate 100 / 8", 12.5),
        ("1,250 + 750", 2000),
        ("12 plus 30", 42),
        ("100 minus 42", 58),
        ("7 times 6", 42),
        ("144 divided by 12", 12),
        ("2 to the power of 10", 1024),
        ("9 squared", 81),
        ("(3 + 4) * 2", 14),
        ("-5 + 10", 5),
        ("10 % 3", 1),
        ("2.5 * 4", 10),
        ("15 × 4", 60),
        ("100 ÷ 4", 25),
    ],
)
def test_computes(question, expected):
    _, result = calculator.calculate(question)
    assert result == pytest.approx(expected)


@pytest.mark.parametrize(
    "question",
    [
        "what is the refund policy?",
        "explain transformers",
        "summarize this document",
        "",
        "   ",
        "42",                       # a bare number is not a calculation
        "what about its limitations?",
        "who is on the 2nd floor",   # a digit, but not arithmetic
        "tell me about section 3",
        "COVID-19 statistics",
    ],
)
def test_refuses_non_arithmetic(question):
    assert not calculator.looks_arithmetic(question)
    with pytest.raises(NotArithmetic):
        calculator.calculate(question)


@pytest.mark.parametrize(
    "hostile",
    [
        "__import__('os').system('echo hi')",
        "open('/etc/passwd').read()",
        "[].__class__.__mro__",
        "(lambda: 1)()",
        "1 if True else 2",
        "x + 1",
        "print(1)",
        "globals()",
    ],
)
def test_refuses_anything_executable(hostile):
    """Nothing that could reach a name, a call or an attribute gets through."""
    with pytest.raises(NotArithmetic):
        calculator.calculate(hostile)


def test_refuses_division_by_zero():
    with pytest.raises(NotArithmetic):
        calculator.calculate("10 / 0")


def test_refuses_absurd_exponent():
    """Guards against a question that would compute for ever."""
    with pytest.raises(NotArithmetic):
        calculator.calculate("9 ** 99999")


def test_formats_readably():
    assert calculator.format_result(200.0) == "200"
    assert calculator.format_result(1234567) == "1,234,567"
    assert calculator.format_result(12.5) == "12.5"


def test_answer_shows_the_expression_used():
    """The reading of the question matters as much as the number."""
    reply = calculator.answer("what is 25% of 800?")
    assert "(25/100)*800" in reply
    assert "200" in reply
