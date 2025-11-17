"""
core.py

Contém a lógica principal de negócio para testar os axiomas 
de um espaço vetorial.
"""

import numpy as np
from typing import Callable, Any, List, Tuple

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