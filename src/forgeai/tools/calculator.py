"""Safe arithmetic via AST whitelisting - never eval()."""
from __future__ import annotations

import ast
import math
import operator as op
import re

_BIN = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.FloorDiv: op.floordiv, ast.Mod: op.mod, ast.Pow: op.pow}
_UN = {ast.UAdd: op.pos, ast.USub: op.neg}
_FUN = {"round": round, "min": min, "max": max, "abs": abs, "sum": lambda *a: sum(a[0]) if len(a) == 1 and isinstance(a[0], (list, tuple)) else sum(a),
        "sqrt": math.sqrt, "ceil": math.ceil, "floor": math.floor}


def _ev(node):
    if isinstance(node, ast.Expression):
        return _ev(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        left, right = _ev(node.left), _ev(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("exponent too large")
        return _BIN[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UN:
        return _UN[type(node.op)](_ev(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUN and not node.keywords:
        return _FUN[node.func.id](*[_ev(a) for a in node.args])
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_ev(e) for e in node.elts]
    raise ValueError(f"unsupported expression element: {type(node).__name__}")


def evaluate(expr: str) -> float:
    expr = str(expr).strip()
    expr = re.sub(r"(?<=\d),(?=\d{3}\b)", "", expr)  # 1,200 -> 1200
    expr = re.sub(r"[$€£₹]|\b(?:USD|EUR|INR|GBP|SGD)\b", "", expr)
    expr = expr.replace("×", "*").replace("^", "**").replace("÷", "/")
    expr = re.sub(r"(?<=[\d)])\s*x\s*(?=[\d(])", "*", expr)  # "3 x 4" -> "3*4"
    if len(expr) > 300:
        raise ValueError("expression too long")
    val = _ev(ast.parse(expr, mode="eval"))
    if isinstance(val, float):
        val = round(val, 6)
        if val.is_integer():
            val = int(val)
    return val


def run(args: dict, ctx=None) -> dict:
    expr = args.get("expression") or args.get("expr") or args.get("input") or ""
    return {"expression": expr, "result": evaluate(expr)}
