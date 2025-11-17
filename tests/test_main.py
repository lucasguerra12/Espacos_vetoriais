import io
import json
import os
import sys
import builtins
from types import SimpleNamespace
from unittest import TestCase, mock

import numpy as np

import main


class TestVectorSpaceTester(TestCase):
    def test_axioms_pass_standard_vector(self):
        shape = (3,)
        add = lambda u, v: u + v
        sm = lambda u, k: k * u
        gen = lambda: np.random.randn(*shape)
        zero = np.zeros(shape)
        tester = main.VectorSpaceTester(add, sm, gen, zero, num_trials=5)
        ok = tester.test()
        self.assertTrue(ok)
        # All axioms should be marked PASSOU
        self.assertTrue(all(v == "PASSOU" for v in tester.results.values()))

    def test_axioms_fail_shifted_addition(self):
        shape = (2,)
        args = SimpleNamespace(operacoes="soma_deslocada", add=None, sm=None, unsafe_eval=False)
        add, sm = main._resolve_operations(args, shape)
        gen = lambda: np.random.randn(*shape)
        zero = np.zeros(shape)
        tester = main.VectorSpaceTester(add, sm, gen, zero, num_trials=5)
        ok = tester.test()
        self.assertFalse(ok)
        # At least one axiom must fail
        self.assertIn("FALHOU", " ".join(tester.results.values()))

    def test_close_tolerance(self):
        shape = (3,)
        add = lambda u, v: u + v
        sm = lambda u, k: k * u
        gen = lambda: np.zeros(shape)
        zero = np.zeros(shape)
        tester = main.VectorSpaceTester(add, sm, gen, zero, num_trials=1, rtol=1e-5, atol=1e-6)
        a = np.array([1.0, 2.0, 3.0])
        b = a + 1e-6
        self.assertTrue(tester._close(a, b))

    def test_check_axiom_exception(self):
        shape = (2,)
        add = lambda u, v: u + v
        sm = lambda u, k: k * u
        gen = lambda: np.zeros(shape)
        zero = np.zeros(shape)
        tester = main.VectorSpaceTester(add, sm, gen, zero, num_trials=1)

        def boom():
            raise ValueError("boom")

        ok = tester.check_axiom(boom, "AXIOMA_TESTE")
        self.assertFalse(ok)
        self.assertIn("FALHOU", tester.results.get("AXIOMA_TESTE", ""))


class TestHelpers(TestCase):
    def test_shape_from_args_vector_and_matrix(self):
        args_v = SimpleNamespace(tipo="vetor", dim=4, rows=0, cols=0)
        self.assertEqual(main._shape_from_args(args_v), (4,))
        args_m = SimpleNamespace(tipo="matriz", dim=0, rows=2, cols=3)
        self.assertEqual(main._shape_from_args(args_m), (2, 3))

    def test_zeros_and_random_gen(self):
        shape = (2, 3)
        z = main._zeros(shape)
        self.assertTrue(np.array_equal(z, np.zeros(shape)))
        gen = main._random_gen(shape)
        a = gen()
        self.assertEqual(a.shape, shape)

    def test_parse_json_arg_ok_and_error(self):
        self.assertEqual(main._parse_json_arg("x", "[1,2]"), [1, 2])
        with self.assertRaises(ValueError):
            main._parse_json_arg("x", "not json")

    def test_ensure_shape(self):
        arr = np.zeros((2, 2))
        main._ensure_shape(arr, (2, 2), "ok")
        with self.assertRaises(ValueError):
            main._ensure_shape(arr, (3, 2), "bad")

    def test_elements_generator_from_list(self):
        elems = [np.array([1, 2]), np.array([3, 4])]
        gen = main._elements_generator_from_list(elems)
        for _ in range(5):
            v = gen()
            self.assertTrue(any(np.array_equal(v, e) for e in elems))

    def test_load_elements_from_args_string_and_file(self):
        shape = (2,)
        elems_json = json.dumps([[0, 0], [1, 2]])
        args = SimpleNamespace(elementos=elems_json, elementos_arquivo=None)
        elems = main._load_elements_from_args(args, shape)
        self.assertEqual(len(elems), 2)

        # From file
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "elems.json")
            with open(p, "w", encoding="utf-8") as f:
                json.dump([[2, 3], [4, 5]], f)
            args2 = SimpleNamespace(elementos=None, elementos_arquivo=p)
            elems2 = main._load_elements_from_args(args2, shape)
            self.assertEqual(len(elems2), 2)

    def test_resolve_operations_presets_and_custom(self):
        shape = (2,)
        # Preset padrao
        args = SimpleNamespace(operacoes="padrao", add=None, sm=None, unsafe_eval=False)
        add, sm = main._resolve_operations(args, shape)
        u = np.array([1.0, 2.0])
        v = np.array([3.0, 4.0])
        self.assertTrue(np.array_equal(add(u, v), np.array([4.0, 6.0])))
        self.assertTrue(np.array_equal(sm(u, 2), np.array([2.0, 4.0])))

        # Invalid preset
        with self.assertRaises(ValueError):
            main._resolve_operations(SimpleNamespace(operacoes="invalido", add=None, sm=None, unsafe_eval=False), shape)

        # Custom lambdas require unsafe_eval
        args_custom = SimpleNamespace(
            operacoes="padrao",
            add="lambda u,v: u-v",
            sm="lambda u,k: u*k + 1",
            unsafe_eval=True,
        )
        add2, sm2 = main._resolve_operations(args_custom, shape)
        self.assertTrue(np.array_equal(add2(u, v), np.array([-2.0, -2.0])))
        self.assertTrue(np.array_equal(sm2(u, 1), np.array([2.0, 3.0])))

        args_unsafe = SimpleNamespace(operacoes="padrao", add="lambda u,v: u+v", sm=None, unsafe_eval=False)
        with self.assertRaises(ValueError):
            main._resolve_operations(args_unsafe, shape)

    def test_prompt_helpers(self):
        # _prompt_yn
        with mock.patch.object(builtins, "input", side_effect=["", "s", "n", "x", "y"]):
            self.assertTrue(main._prompt_yn("?", True))   # default True
            self.assertTrue(main._prompt_yn("?", False))  # 's'
            self.assertFalse(main._prompt_yn("?", True))  # 'n'
            # 'x' then 'y'
            self.assertTrue(main._prompt_yn("?", False))

        # _prompt_int
        with mock.patch.object(builtins, "input", side_effect=["", "10"]):
            self.assertEqual(main._prompt_int("n", 5), 5)
            self.assertEqual(main._prompt_int("n"), 10)

        # _prompt_float
        with mock.patch.object(builtins, "input", side_effect=["", "3,14", "2.5"]):
            self.assertAlmostEqual(main._prompt_float("x", 1.5), 1.5)
            self.assertAlmostEqual(main._prompt_float("x"), 3.14)
            self.assertAlmostEqual(main._prompt_float("x"), 2.5)

        # _parse_numbers
        nums = main._parse_numbers("1, 2; 3 4")
        self.assertEqual(nums, [1.0, 2.0, 3.0, 4.0])

    def test_input_vector_and_matrix(self):
        # _input_vector: wrong length then correct
        with mock.patch.object(builtins, "input", side_effect=["1,2", "1,2,3"]):
            v = main._input_vector(3)
            self.assertTrue(np.array_equal(v, np.array([1.0, 2.0, 3.0])))

        # _input_matrix 2x2: wrong then correct
        inputs = [
            "1,2,3",  # wrong cols
            "1,2",    # row 1 ok
            "4",      # wrong cols row 2
            "3,4",    # row 2 ok
        ]
        with mock.patch.object(builtins, "input", side_effect=inputs):
            m = main._input_matrix(2, 2)
            self.assertTrue(np.array_equal(m, np.array([[1.0, 2.0], [3.0, 4.0]])))

    def test_run_interactive_minimal_path(self):
        # Sequence to accept defaults where possible and finish
        # tipo: vetor, dim: blank->3, ensaios: 1, seed blank, rtol blank, atol blank,
        # zero? n, elementos? n, op: blank->1, custom? n
        inputs = [
            "vetor",  # tipo
            "",       # dim default 3
            "1",      # ensaios
            "",       # seed
            "",       # rtol
            "",       # atol
            "n",      # zero?
            "n",      # elementos?
            "",       # op default 1
            "n",      # custom ops?
        ]
        with mock.patch.object(builtins, "input", side_effect=inputs):
            code = main._run_interactive()
            self.assertIn(code, (0, 1))  # both outcomes acceptable, shouldn't raise


class TestMainCLI(TestCase):
    def test_cli_non_interactive_standard(self):
        argv = [
            "main.py",
            "--tipo",
            "vetor",
            "--dim",
            "2",
            "--ensaios",
            "1",
            "--operacoes",
            "padrao",
            "--seed",
            "0",
        ]
        with mock.patch.object(sys, "argv", argv):
            code = main.main_cli()
            self.assertEqual(code, 0)

    def test_cli_invalid_operations(self):
        argv = [
            "main.py",
            "--tipo",
            "vetor",
            "--dim",
            "2",
            "--operacoes",
            "invalido",
        ]
        with mock.patch.object(sys, "argv", argv):
            with self.assertRaises(SystemExit) as cm:
                main.main_cli()
            self.assertEqual(cm.exception.code, 2)
