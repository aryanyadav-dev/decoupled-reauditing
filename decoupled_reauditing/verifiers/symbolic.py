import re
from dataclasses import dataclass

from decoupled_reauditing.utils import parse_final_answer


_EXPR_RE = re.compile(r"(?<!\w)([-+]?\d+(?:\.\d+)?(?:\s*[-+*/]\s*[-+]?\d+(?:\.\d+)?)+)\s*=\s*([-+]?\d+(?:\.\d+)?)")


def _safe_eval(expr: str):
    if not re.fullmatch(r"[\d\s+\-*/().]+", expr):
        return None
    try:
        return eval(expr, {"__builtins__": {}}, {})
    except Exception:
        return None


@dataclass
class SymbolicChecker:
    name: str = "symbolic_checker"

    def accepts(self, problem, trace, context=None) -> bool:
        stated = parse_final_answer(trace)
        if stated is None:
            return False
        equations = list(_EXPR_RE.finditer(trace))
        if not equations:
            return False
        for match in equations:
            lhs, rhs = match.group(1), match.group(2)
            val = _safe_eval(lhs)
            if val is None or abs(float(val) - float(rhs)) > 1e-6:
                return False
        last_rhs = float(equations[-1].group(2))
        return abs(last_rhs - float(stated)) < 1e-6

