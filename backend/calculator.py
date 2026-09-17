"""Arithmetic, worked out here rather than by the model.

Language models are unreliable at arithmetic and every call costs quota, so a
question that is only a sum is answered locally. The evaluation harness gets
a question with a knowable right answer out of it, too.

`eval()` is not used anywhere in this file. The expression is parsed to a
syntax tree and walked with an explicit whitelist of node types, so there is
no path from a typed question to attribute access, a name lookup, or a call.
"""

import ast
import operator
import re

# Only these survive the walk. Anything else - a name, a call, a subscript,
# an attribute - raises, which is what keeps this safe rather than clever.
BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# Guards against a question that parses but should not be computed here.
MAX_EXPONENT = 1000

PHRASES = [
    # "25% of 800" is the one worth handling: it is the commonest calculation
    # people ask in words, and it has no operator at all.
    (re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|percent(?:age)?)\s+of\b"), r"(\1/100)*"),
    (re.compile(r"\bplus\b|\badd(?:ed to)?\b"), "+"),
    (re.compile(r"\bminus\b|\bless\b|\bsubtract(?:ed from)?\b"), "-"),
    (re.compile(r"\btimes\b|\bmultiplied by\b|\bmultiply by\b"), "*"),
    (re.compile(r"\bdivided by\b|\bover\b"), "/"),
    (re.compile(r"\bto the power of\b|\braised to\b|\^"), "**"),
    (re.compile(r"\bsquared\b"), "**2"),
    (re.compile(r"\bcubed\b"), "**3"),
]

# Wording around the sum that carries no arithmetic.
NOISE = re.compile(
    r"^\s*(?:what(?:'s| is)?|whats|how much is|calculate|compute|work out|"
    r"tell me|please|can you)\b[\s:,]*",
    re.IGNORECASE,
)

# What an expression may consist of once the words are gone.
EXPRESSION_ONLY = re.compile(r"^[\d\s+\-*/%().]+$")


class NotArithmetic(ValueError):
    """The text is not something this module should be answering."""


def to_expression(text: str) -> str:
    """Turn a question into a bare arithmetic expression, or raise.

    Raising rather than guessing matters: this decides whether a question is
    routed away from the language model entirely, so a false positive means a
    real question gets answered with a number.
    """
    working = NOISE.sub("", (text or "").strip())
    working = working.rstrip("?.! ").strip()
    working = working.replace("×", "*").replace("÷", "/").replace("−", "-")
    # Thousands separators, but only between digits - not the comma in a list.
    working = re.sub(r"(?<=\d),(?=\d{3}\b)", "", working)
    working = working.lower()

    for pattern, replacement in PHRASES:
        working = pattern.sub(replacement, working)

    # A bare "x" between numbers means multiply; elsewhere it is a variable.
    working = re.sub(r"(?<=[\d\s)])x(?=[\s(\d])", "*", working)
    working = working.strip()

    if not working or not EXPRESSION_ONLY.match(working):
        raise NotArithmetic(f"not an arithmetic expression: {text!r}")
    if not re.search(r"\d", working):
        raise NotArithmetic("no numbers to work with")
    if not re.search(r"[+\-*/%]", working):
        # A lone number is not a calculation; it is probably a fragment of
        # something the model should handle.
        raise NotArithmetic("no operator")
    if re.search(r"\d\s+\d", working):
        # Two numbers with nothing between them - "room 12 30" rather than a
        # sum. Refusing beats inventing an operator.
        raise NotArithmetic("two numbers with no operator between them")

    # Safe now that nothing but digits, operators and brackets remain, and
    # it is the form shown back to the user.
    return re.sub(r"\s+", "", working)


def _evaluate(node):
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise NotArithmetic(f"unsupported value: {node.value!r}")
        return node.value

    if isinstance(node, ast.UnaryOp):
        handler = UNARY_OPERATORS.get(type(node.op))
        if handler is None:
            raise NotArithmetic(f"unsupported operator: {type(node.op).__name__}")
        return handler(_evaluate(node.operand))

    if isinstance(node, ast.BinOp):
        handler = BINARY_OPERATORS.get(type(node.op))
        if handler is None:
            raise NotArithmetic(f"unsupported operator: {type(node.op).__name__}")
        left, right = _evaluate(node.left), _evaluate(node.right)

        # Without this, 9**9**9 hangs the app computing a number nobody asked
        # for and nobody can read.
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise NotArithmetic("exponent too large")
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
            raise NotArithmetic("division by zero")
        return handler(left, right)

    raise NotArithmetic(f"unsupported syntax: {type(node).__name__}")


def looks_arithmetic(text: str) -> bool:
    """True when this question is a sum, and can be answered without a model."""
    try:
        to_expression(text)
    except NotArithmetic:
        return False
    return True


def calculate(text: str) -> tuple[str, float]:
    """Work out the answer. Returns (expression, result), or raises.

    The expression comes back too, so the UI can show what was actually
    computed - the reading of the question matters as much as the number.
    """
    expression = to_expression(text)
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise NotArithmetic(f"could not parse {expression!r}") from exc

    result = _evaluate(tree)
    if isinstance(result, complex) or result != result or result in (float("inf"), float("-inf")):
        raise NotArithmetic("the result is not a finite number")
    return expression, result


def format_result(value: float) -> str:
    """Render a number the way someone would write it."""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, int):
        return f"{value:,}"
    # Enough places to be useful, without a trail of floating-point noise.
    return f"{round(value, 10):,}".rstrip("0").rstrip(".")


def answer(text: str) -> str:
    """The full response for a calculation, ready to show."""
    expression, value = calculate(text)
    return f"{expression} = **{format_result(value)}**"
