"""
ui_helpers.py

Funções auxiliares para lidar com a entrada do utilizador (prompts)
no terminal, garantindo a validação dos dados.
"""

import numpy as np
from typing import List

# -------------------- Funções de Prompt (Modo Interativo) --------------------

def _prompt_yn(msg: str, default: bool = False) -> bool:
    """Prompt para resposta sim/não com default, aceita variações ('s','sim','y')."""
    suf = "[S/n]" if default else "[s/N]"
    while True:
        resp = input(f"{msg} {suf}: ").strip().lower()
        if not resp:
            return default
        if resp in ("s", "sim", "y", "yes"):
            return True
        if resp in ("n", "nao", "não", "no"):
            return False
        print("Por favor, responda com s/n.")


def _prompt_int(msg: str, default: int | None = None) -> int:
    """Lê inteiro com validação e fallback opcional para valor default."""
    while True:
        txt = input(f"{msg}{' ['+str(default)+']' if default is not None else ''}: ").strip()
        if not txt and default is not None:
            return default
        try:
            return int(txt)
        except Exception:
            print("Valor inválido. Digite um número inteiro.")


def _prompt_float(msg: str, default: float | None = None) -> float:
    """Lê float permitindo vírgula ou ponto como separador decimal."""
    while True:
        txt = input(f"{msg}{' ['+str(default)+']' if default is not None else ''}: ").strip()
        if not txt and default is not None:
            return default
        try:
            return float(txt.replace(',', '.'))
        except Exception:
            print("Valor inválido. Digite um número (use ponto para decimais).")


def _parse_numbers(line: str) -> List[float]:
    """Extrai lista de floats de uma linha separada por espaços, vírgulas ou ponto-e-vírgula."""
    parts = [p for p in line.replace(';', ' ').replace(',', ' ').split() if p]
    return [float(p) for p in parts]


def _input_vector(dim: int) -> np.ndarray:
    """Lê um vetor de dimensão 'dim', repetindo até o número correto de valores."""
    while True:
        line = input(f"Informe {dim} números (separados por espaço ou vírgula): ")
        try:
            vals = _parse_numbers(line)
            if len(vals) != dim:
                print(f"Forneça exatamente {dim} valores.")
                continue
            return np.array(vals, dtype=float)
        except Exception as e:
            print(f"Entrada inválida: {e}")


def _input_matrix(rows: int, cols: int) -> np.ndarray:
    """Lê uma matriz linha a linha garantindo 'cols' valores por linha."""
    mat = np.zeros((rows, cols), dtype=float)
    for r in range(rows):
        while True:
            line = input(f"Linha {r+1}/{rows}: informe {cols} números: ")
            try:
                vals = _parse_numbers(line)
                if len(vals) != cols:
                    print(f"Forneça exatamente {cols} valores.")
                    continue
                mat[r, :] = vals
                break
            except Exception as e:
                print(f"Entrada inválida: {e}")
    return mat