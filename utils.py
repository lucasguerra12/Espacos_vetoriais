"""
utils.py

Funções auxiliares para manipulação de dados, como parsing de JSON,
definição de shapes e resolução de operações (presets).
"""

import json
import numpy as np
from typing import Callable, Any, List, Tuple

# -------------------- CLI & Utilidades --------------------

def _shape_from_args(args) -> Tuple[int, ...]:
    """Retorna o shape lógico a partir dos argumentos.

    vetor => (dim,), matriz => (rows, cols)
    """
    if args.tipo == "vetor":
        return (args.dim,)
    else:
        return (args.rows, args.cols)


def _zeros(shape: Tuple[int, ...]):
    """Cria um elemento zero (vetor/matriz nulo) no shape fornecido."""
    return np.zeros(shape)


def _random_gen(shape: Tuple[int, ...]):
    """Retorna função geradora de elementos aleatórios com distribuição normal."""
    def gen():
        return np.random.randn(*shape)
    return gen


def _parse_json_arg(name: str, value: str):
    """Converte string JSON e levanta erro amigável incluindo o nome do parâmetro."""
    try:
        return json.loads(value)
    except Exception as e:
        raise ValueError(f"Não foi possível interpretar {name} como JSON: {e}")


def _ensure_shape(arr: np.ndarray, shape: Tuple[int, ...], label: str):
    """Valida se 'arr' tem o shape esperado, senão lança ValueError com rótulo."""
    if tuple(arr.shape) != tuple(shape):
        raise ValueError(
            f"Elemento '{label}' possui shape {arr.shape}, esperado {shape}."
        )


def _elements_generator_from_list(elems: List[np.ndarray]):
    """Retorna gerador que amostra uniformemente da lista de elementos fornecida."""
    def gen():
        idx = np.random.randint(0, len(elems))
        return elems[idx]
    return gen


def _load_elements_from_args(args, shape: Tuple[int, ...]) -> List[np.ndarray]:
    """Carrega elementos a partir de --elementos (string JSON) ou --elementos-arquivo.

    Cada item deve ter o mesmo shape; converte para np.array(dtype=float).
    """
    elems: List[np.ndarray] = []
    if args.elementos:
        data = _parse_json_arg("--elementos", args.elementos)
        if isinstance(data, list):
            for i, item in enumerate(data):
                arr = np.array(item, dtype=float)
                _ensure_shape(arr, shape, f"elementos[{i}]")
                elems.append(arr)
        else:
            raise ValueError("--elementos deve ser uma lista JSON de vetores/matrizes.")
    if args.elementos_arquivo:
        with open(args.elementos_arquivo, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for i, item in enumerate(data):
                arr = np.array(item, dtype=float)
                _ensure_shape(arr, shape, f"elementos_arquivo[{i}]")
                elems.append(arr)
        else:
            raise ValueError(
                "--elementos-arquivo deve conter uma lista JSON de vetores/matrizes."
            )
    return elems


def _resolve_operations(args, shape: Tuple[int, ...]):
    # Presets seguros (cada ramo define add, sm). 'bias' usado apenas em soma_deslocada.
    if args.operacoes == "padrao":
        add = lambda u, v: u + v
        sm = lambda u, k: k * u
        bias = np.zeros(shape)
    elif args.operacoes == "soma_deslocada":
        # u ⊕ v = u + v + c (com c != 0)
        bias = np.ones(shape)
        add = lambda u, v, b=bias: u + v + b
        sm = lambda u, k: k * u
    elif args.operacoes == "escalar_estranho":
        # k ⊗ u = (k+1) * u
        add = lambda u, v: u + v
        sm = lambda u, k: (k + 1.0) * u
        bias = np.zeros(shape)
    else:
        raise ValueError("Preset de operações inválido.")

    # Operações customizadas (inseguras): precisam de --unsafe-eval
    if args.add or args.sm:
        if not args.unsafe_eval:
            raise ValueError(
                "Para usar --add/--sm personalizados, adicione também --unsafe-eval."
            )
        safe_globals = {"np": np}  # exposição controlada de NumPy
        if args.add:
            try:
                add = eval(args.add, safe_globals)
            except Exception as e:
                raise ValueError(f"Erro ao avaliar --add: {e}")
        if args.sm:
            try:
                sm = eval(args.sm, safe_globals)
            except Exception as e:
                raise ValueError(f"Erro ao avaliar --sm: {e}")

    return add, sm


# Prompt inicial para operações (deve ficar antes de main_cli)
def _prompt_operation_startup() -> str:
    """Prompt inicial para escolher operação quando script é executado sem --operacoes.

    Retorna uma string do conjunto {"padrao", "soma_deslocada", "escalar_estranho"}.
    """
    print("== Seleção de Operações ==")
    print("Escolha o conjunto de operações antes de continuar:")
    print("  1) padrao         (u+v, k*u)")
    print("  2) soma_deslocada (u+v+1)")
    print("  3) escalar_estranho ((k+1)*u)")
    while True:
        choice = input("Opção [1/2/3] [1]: ").strip() or "1"
        mapping = {"1": "padrao", "2": "soma_deslocada", "3": "escalar_estranho"}
        if choice in mapping:
            return mapping[choice]
        print("Valor inválido. Digite 1, 2 ou 3.")