"""main.py

Ferramenta para testar (por amostragem probabilística) se um conjunto de vetores
ou matrizes com operações de adição e multiplicação escalar satisfaz os 8 axiomas
de espaço vetorial. O usuário pode escolher entre três modos de operação:

1. Modo direto (não interativo): gera elementos aleatórios, ou lê listas via JSON.
2. Modo interativo completo (--interativo): pergunta passo a passo dimensões, tolerâncias,
     elementos, zero e operações (inclui opção para lambdas customizadas).
3. Modo de entrada manual simplificada (--entrada-manual): foca em digitar somente
     números dos vetores/matrizes, zero e escolher operação.

Arquitetura rápida:
- VectorSpaceTester: encapsula a lógica dos 8 axiomas e registra resultados.
- Helpers de parsing: criação de elementos, comparação numérica com tolerâncias.
- Presets de operações: 'padrao', 'soma_deslocada' e 'escalar_estranho' permitem observar
    sucesso/falha de axiomas intencionalmente.
- Funções de prompt: abstraem entrada robusta (inteiros, floats, vetores, matrizes).

Decisões principais:
- Usa NumPy para geração e comparação (allclose) por estabilidade numérica.
- Inverso aditivo calculado como sm(u, -1) assumindo multiplicação escalar padrão.
    (Nos presets que deformam a soma ou escalar, isso provoca falhas esperadas.)
- Comparação _close tenta degradar para igualdade simples se não for array NumPy.

Limitações:
- Probabilístico: não garante prova formal; aumentar --ensaios para maior confiança.
- Presets não substituem análise teórica; servem para demonstrar quebra de axiomas.

"""

import argparse
import json
from typing import Callable, Any, List, Tuple
from types import SimpleNamespace
import numpy as np


class VectorSpaceTester:
    """
    Testa probabilisticamente se um conjunto V com operações de adição
    e multiplicação escalar forma um espaço vetorial.
    """

    def __init__(
        self,
        add_func: Callable[[Any, Any], Any],
        scalar_mult_func: Callable[[Any, float], Any],
        element_generator: Callable[[], Any],
        zero_element: Any,
        num_trials: int = 100,
        rtol: float = 1e-7,
        atol: float = 1e-8,
    ):
        """
        Inicializa o testador.

        Args:
            add_func (function): Uma função (lambda) que recebe dois elementos
                                 de V e retorna sua soma. Ex: lambda u, v: u + v
            scalar_mult_func (function): Uma função (lambda) que recebe um elemento
                                         de V e um escalar (k) e retorna o produto.
                                         Ex: lambda u, k: u * k
            element_generator (function): Uma função (lambda) que não recebe argumentos
                                          e retorna um elemento aleatório de V.
                                          Ex: lambda: np.random.rand(3) para R^3
            zero_element (object): O elemento neutro (vetor/matriz nula) do conjunto V.
                                   Ex: np.zeros(3) para R^3
            num_trials (int): O número de testes aleatórios para cada axioma.
            rtol (float): Tolerância relativa para comparações numéricas.
            atol (float): Tolerância absoluta para comparações numéricas.
        """
        self.add = add_func
        self.sm = scalar_mult_func
        self.gen = element_generator
        self.zero = zero_element
        self.num_trials = int(num_trials)
        self.rtol = float(rtol)
        self.atol = float(atol)
        self.results = {}

        # Para o Axioma 4, assumimos que o inverso de 'u' é '-1 * u'
        self.inv = lambda u: self.sm(u, -1)

    def _close(self, a: Any, b: Any) -> bool:
        """Compara dois elementos considerando tolerâncias rtol/atol.

        Tenta primeiro via numpy.allclose. Caso 'a' ou 'b' não sejam arrays NumPy
        (ou provoquem exceções), converte para np.array e aplica um critério manual;
        no último caso, recorre à igualdade direta.
        """
        try:
            return np.allclose(a, b, rtol=self.rtol, atol=self.atol)
        except Exception:
            try:
                # Critério manual: max(|a-b|) <= atol + rtol*max(|b|)
                return abs(np.array(a) - np.array(b)).max() <= (
                    self.atol + self.rtol * abs(np.array(b)).max()
                )
            except Exception:
                # Último recurso para tipos não numéricos.
                return a == b

    def check_axiom(self, axiom_func, axiom_name):
        """Executa 'axiom_func' num loop de 'num_trials' e registra PASSOU/FALHOU.

        Se qualquer tentativa falhar, marca o axioma como FALHOU e interrompe cedo.
        Captura exceções para registrar erros de dimensão ou lógica.
        """
        try:
            for _ in range(self.num_trials):
                if not axiom_func():
                    self.results[axiom_name] = "FALHOU"
                    return False
            self.results[axiom_name] = "PASSOU"
            return True
        except Exception as e:
            # Captura erros de dimensão (ex: somar R^2 com R^3)
            self.results[axiom_name] = f"FALHOU (Erro: {e})"
            return False

    # --- Definição dos 8 Axiomas ---

    def a1_commutative(self):
        # u + v == v + u
        u, v = self.gen(), self.gen()
        return self._close(self.add(u, v), self.add(v, u))

    def a2_associative_add(self):
        # (u+v)+w == u+(v+w)
        u, v, w = self.gen(), self.gen(), self.gen()
        return self._close(self.add(self.add(u, v), w), self.add(u, self.add(v, w)))

    def a3_zero_element(self):
        # u + 0 == u
        u = self.gen()
        return self._close(self.add(u, self.zero), u)

    def a4_inverse_element(self):
        # u + (-u) == 0 (usa inverso como -1 * u)
        u = self.gen()
        neg_u = self.inv(u)
        return self._close(self.add(u, neg_u), self.zero)

    def a5_distributive_scalar_vector(self):
        # k*(u+v) == k*u + k*v
        k = float(np.random.randn())
        u, v = self.gen(), self.gen()
        lhs = self.sm(self.add(u, v), k)
        rhs = self.add(self.sm(u, k), self.sm(v, k))
        return self._close(lhs, rhs)

    def a6_distributive_scalar_scalar(self):
        # (k+m)*u == k*u + m*u
        k, m = float(np.random.randn()), float(np.random.randn())
        u = self.gen()
        lhs = self.sm(u, k + m)
        rhs = self.add(self.sm(u, k), self.sm(u, m))
        return self._close(lhs, rhs)

    def a7_associative_scalar(self):
        # (k*m)*u == k*(m*u)
        k, m = float(np.random.randn()), float(np.random.randn())
        u = self.gen()
        lhs = self.sm(u, k * m)
        rhs = self.sm(self.sm(u, m), k)
        return self._close(lhs, rhs)

    def a8_identity_scalar(self):
        # 1*u == u
        u = self.gen()
        return self._close(self.sm(u, 1), u)

    # --- Função Principal ---

    def test(self) -> bool:
        """Executa os 8 axiomas, imprime resultados e retorna True se todos PASSARAM.

        Estratégia: avalia cada axioma com 'num_trials' amostras aleatórias.
        Falhas em qualquer tentativa interrompem o teste daquele axioma (curto-circuito).
        """
        print(
            f"Iniciando teste de Espaço Vetorial ({self.num_trials} tentativas por axioma) "
            f"com rtol={self.rtol}, atol={self.atol}..."
        )

        axioms_to_test: List[Tuple[Callable[[], bool], str]] = [
            (self.a1_commutative, "A1: Comutatividade da Adição (u + v = v + u)"),
            (self.a2_associative_add, "A2: Associatividade da Adição ((u+v)+w = u+(v+w))"),
            (self.a3_zero_element, "A3: Elemento Neutro da Adição (u + 0 = u)"),
            (self.a4_inverse_element, "A4: Elemento Inverso da Adição (u + (-u) = 0)"),
            (self.a5_distributive_scalar_vector, "A5: Distributividade (k*(u+v) = k*u + k*v)"),
            (self.a6_distributive_scalar_scalar, "A6: Distributividade ((k+m)*u = k*u + m*u)"),
            (self.a7_associative_scalar, "A7: Associatividade Escalar ((k*m)*u = k*(m*u))"),
            (self.a8_identity_scalar, "A8: Identidade Escalar (1 * u = u)"),
        ]

        all_passed = True
        for func, name in axioms_to_test:
            if not self.check_axiom(func, name):
                all_passed = False

        print("\n--- Resultados Detalhados ---")
        for name, status in self.results.items():
            print(f"{name}: {status}")

        print("\n--- Conclusão ---")
        if all_passed:
            print("VEREDITO: O conjunto com as operações dadas PARECE ser um Espaço Vetorial.")
        else:
            print(
                "VEREDITO: O conjunto com as operações dadas NÃO é um Espaço Vetorial."
            )

        return all_passed


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

# -------------------- Modo Interativo --------------------

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
    elems: List[np.ndarray] = []
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


# -------------------- Modo Interativo --------------------

def _prompt_yn(msg: str, default: bool = False) -> bool:
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
    while True:
        txt = input(f"{msg}{' ['+str(default)+']' if default is not None else ''}: ").strip()
        if not txt and default is not None:
            return default
        try:
            return int(txt)
        except Exception:
            print("Valor inválido. Digite um número inteiro.")


def _prompt_float(msg: str, default: float | None = None) -> float:
    while True:
        txt = input(f"{msg}{' ['+str(default)+']' if default is not None else ''}: ").strip()
        if not txt and default is not None:
            return default
        try:
            return float(txt.replace(',', '.'))
        except Exception:
            print("Valor inválido. Digite um número (use ponto para decimais).")


def _parse_numbers(line: str) -> List[float]:
    parts = [p for p in line.replace(';', ' ').replace(',', ' ').split() if p]
    return [float(p) for p in parts]


def _input_vector(dim: int) -> np.ndarray:
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
