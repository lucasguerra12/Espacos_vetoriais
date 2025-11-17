"""
main.py

Ponto de entrada principal da ferramenta de teste de Espaço Vetorial.

Este ficheiro é responsável por:
1. Definir a interface de linha de comando (CLI) com argparse.
2. Coordenar os diferentes modos de execução (interativo, manual, direto).
3. Importar a lógica de negócio (core), utilitários (utils) e 
   ajudantes de UI (ui_helpers) para executar os testes.
"""

import argparse
import sys
from types import SimpleNamespace
import numpy as np

# Imports dos módulos refatorados
from core import VectorSpaceTester
from ui_helpers import (
    _prompt_yn, _prompt_int, _prompt_float, _input_vector, _input_matrix
)
from utils import (
    _shape_from_args, _zeros, _random_gen, _parse_json_arg,
    _ensure_shape, _elements_generator_from_list,
    _load_elements_from_args, _resolve_operations,
    _prompt_operation_startup
)

# -------------------- Modos de Execução --------------------

def _run_interactive() -> int:
    print("=== Teste de Espaço Vetorial (Modo Interativo) ===")  # banner de modo

    # Tipo e dimensões
    tipo = input("Escolha o tipo [vetor/matriz] [vetor]: ").strip().lower() or "vetor"
    if tipo not in ("vetor", "matriz"):
        raise ValueError("Tipo inválido. Use 'vetor' ou 'matriz'.")

    if tipo == "vetor":
        dim = _prompt_int("Dimensão do vetor", 3)
        shape = (dim,)
    else:
        rows = _prompt_int("Linhas", 2)
        cols = _prompt_int("Colunas", 2)
        shape = (rows, cols)

    ensaios = _prompt_int("Número de tentativas por axioma", 100)
    seed_val = input("Semente (vazio para aleatório): ").strip()
    if seed_val:
        np.random.seed(int(seed_val))

    rtol = _prompt_float("Tolerância relativa (rtol)", 1e-7)
    atol = _prompt_float("Tolerância absoluta (atol)", 1e-8)

    # Elemento zero
    if _prompt_yn("Deseja informar o elemento zero?", False):
        if tipo == "vetor":
            zero = _input_vector(shape[0])
        else:
            zero = _input_matrix(shape[0], shape[1])
    else:
        zero = _zeros(shape)

    # Elementos
    elems = []
    if _prompt_yn("Deseja informar uma lista de elementos?", False):
        n = _prompt_int("Quantos elementos deseja informar?", 3)
        for i in range(n):
            print(f"Elemento {i+1}/{n}")
            if tipo == "vetor":
                arr = _input_vector(shape[0])
            else:
                arr = _input_matrix(shape[0], shape[1])
            elems.append(arr)

    # Operações
    print("Escolha o conjunto de operações:")
    print("  1) padrao (u+v, k*u)")
    print("  2) soma_deslocada (u+v+1)")
    print("  3) escalar_estranho ((k+1)*u)")
    op_choice = input("Opção [1/2/3] [1]: ").strip() or "1"
    op_map = {"1": "padrao", "2": "soma_deslocada", "3": "escalar_estranho"}
    operacoes = op_map.get(op_choice, "padrao")

    add_str = None
    sm_str = None
    unsafe = False
    if _prompt_yn("Deseja definir lambdas customizadas para as operações?", False):
        print("ATENÇÃO: Isto irá avaliar código Python digitado por você.")
        unsafe = _prompt_yn("Confirmar uso de avaliação insegura?", False)
        if unsafe:
            add_str = input("Lambda para adição (ex.: lambda u,v: u+v): ").strip() or None
            sm_str = input("Lambda para escalar (ex.: lambda u,k: k*u): ").strip() or None
        else:
            print("Operações customizadas canceladas. Usando preset selecionado.")

    # Construir args mínimos para reutilizar utilitários
    args_ns = SimpleNamespace(
        operacoes=operacoes, add=add_str, sm=sm_str, unsafe_eval=unsafe
    )
    add, sm = _resolve_operations(args_ns, shape)

    # Gerador
    if elems:
        gen = _elements_generator_from_list(elems)
    else:
        gen = _random_gen(shape)

    tester = VectorSpaceTester(add, sm, gen, zero, num_trials=ensaios, rtol=rtol, atol=atol)
    ok = tester.test()
    return 0 if ok else 1

def _run_manual_input(args) -> int:
    """Fluxo reduzido de entrada manual: pergunta dimensões (se não fornecidas), elemento zero, lista de elementos e operação.

    Usa 'args' somente para pegar flags já informadas; ignora gerador aleatório padrão e sempre usa elementos fornecidos/manual.
    """
    print("=== Espaço Vetorial (Entrada Manual) ===")  # banner de modo

    # Tipo
    tipo = args.tipo if args.tipo else (input("Tipo [vetor/matriz] [vetor]: ").strip() or "vetor")
    if tipo not in ("vetor", "matriz"):
        raise ValueError("Tipo inválido. Use 'vetor' ou 'matriz'.")

    # Dimensões
    if tipo == "vetor":
        dim = args.dim if args.dim else _prompt_int("Dimensão do vetor", 3)
        shape = (dim,)
    else:
        rows = args.rows if args.rows else _prompt_int("Linhas", 2)
        cols = args.cols if args.cols else _prompt_int("Colunas", 2)
        shape = (rows, cols)

    ensaios = args.ensaios if hasattr(args, 'ensaios') and args.ensaios else _prompt_int("Número de tentativas por axioma", 50)

    # Zero manual
    if _prompt_yn("Informar elemento zero?", True):
        if tipo == "vetor":
            zero = _input_vector(shape[0])
        else:
            zero = _input_matrix(shape[0], shape[1])
    else:
        zero = _zeros(shape)

    # Lista de elementos
    elems = []
    n_elems = _prompt_int("Quantos elementos deseja informar?", 3)
    for i in range(n_elems):
        print(f"Elemento {i+1}/{n_elems}")
        if tipo == "vetor":
            arr = _input_vector(shape[0])
        else:
            arr = _input_matrix(shape[0], shape[1])
        elems.append(arr)

    # Operação
    operacoes = _prompt_operation_startup()
    args_ns = SimpleNamespace(operacoes=operacoes, add=None, sm=None, unsafe_eval=False)
    add, sm = _resolve_operations(args_ns, shape)

    gen = _elements_generator_from_list(elems)

    rtol = args.rtol if hasattr(args, 'rtol') and args.rtol is not None else 1e-7
    atol = args.atol if hasattr(args, 'atol') and args.atol is not None else 1e-8
    tester = VectorSpaceTester(add, sm, gen, zero, num_trials=ensaios, rtol=rtol, atol=atol)
    ok = tester.test()
    return 0 if ok else 1

def main_cli():
    parser = argparse.ArgumentParser(
        description=(
            "Teste por amostragem dos axiomas de espaço vetorial para vetores ou matrizes."
        )
    )

    # Espaço
    parser.add_argument("--tipo", choices=["vetor", "matriz"], default="vetor")
    parser.add_argument("--dim", type=int, default=3, help="Dimensão do vetor (se tipo=vetor)")
    parser.add_argument("--rows", type=int, default=2, help="Linhas (se tipo=matriz)")
    parser.add_argument("--cols", type=int, default=2, help="Colunas (se tipo=matriz)")

    # Amostragem
    parser.add_argument("--ensaios", type=int, default=100, help="Número de tentativas por axioma")
    parser.add_argument("--seed", type=int, default=None, help="Semente do gerador aleatório")

    # Tolerâncias
    parser.add_argument("--rtol", type=float, default=1e-7, help="Tolerância relativa")
    parser.add_argument("--atol", type=float, default=1e-8, help="Tolerância absoluta")

    # Elementos
    parser.add_argument(
        "--elementos",
        type=str,
        default=None,
        help="Lista JSON de elementos (vetores/matrizes) a serem amostrados",
    )
    parser.add_argument(
        "--elementos-arquivo",
        type=str,
        default=None,
        help="Caminho para arquivo JSON com lista de elementos",
    )
    parser.add_argument(
        "--zero",
        type=str,
        default=None,
        help="Elemento zero em JSON (se omitido, usa zeros no shape informado)",
    )

    # Operações
    parser.add_argument(
        "--operacoes",
        choices=["padrao", "soma_deslocada", "escalar_estranho"],
        default=None,
        help="Conjunto de operações predefinidas (se omitido será perguntado ao iniciar)",
    )
    parser.add_argument("--add", type=str, default=None, help="Lambda customizada para adição: ex. 'lambda u,v: u+v' ")
    parser.add_argument("--sm", type=str, default=None, help="Lambda customizada para mult. escalar: ex. 'lambda u,k: k*u' ")
    parser.add_argument("--unsafe-eval", action="store_true", help="Permite avaliar lambdas customizadas (use com cautela)")
    parser.add_argument("--interativo", action="store_true", help="Modo interativo: responda perguntas no terminal")
    parser.add_argument("--entrada-manual", action="store_true", help="Modo de entrada manual simplificado (apenas números e operações)")

    args = parser.parse_args()

    # Modo interativo completo
    if args.interativo:
        return _run_interactive()

    if args.entrada_manual:
        return _run_manual_input(args)

    # Escolher operação imediatamente ao iniciar se não fornecida
    if args.operacoes is None and not (args.add or args.sm):
        args.operacoes = _prompt_operation_startup()
        # Se o usuário não forneceu elementos por JSON/arquivo, ofereça entrada manual
        if not args.elementos and not args.elementos_arquivo:
            if _prompt_yn("Deseja digitar manualmente os vetores/matrizes?", True):
                return _run_manual_input(args)

    # Semente
    if args.seed is not None:
        np.random.seed(args.seed)

    # Shape e zero
    shape = _shape_from_args(args)

    if args.zero:
        zero_data = _parse_json_arg("--zero", args.zero)
        zero = np.array(zero_data, dtype=float)
        _ensure_shape(zero, shape, "zero")
    else:
        zero = _zeros(shape)

    # Elementos
    elems = _load_elements_from_args(args, shape)
    if elems:
        gen = _elements_generator_from_list(elems)
    else:
        gen = _random_gen(shape)

    # Operações
    add, sm = _resolve_operations(args, shape)

    tester = VectorSpaceTester(add, sm, gen, zero, num_trials=args.ensaios, rtol=args.rtol, atol=args.atol)

    ok = tester.test()
    # Código de saída útil para automação
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        exit_code = main_cli()
        raise SystemExit(exit_code)
    except Exception as e:
        print(f"Erro: {e}")
        raise SystemExit(2)