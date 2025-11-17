# Teste de Espaço Vetorial (CLI)

Este utilitário em Python testa, por amostragem, os 8 axiomas de espaço vetorial para conjuntos de vetores ou matrizes, com operações configuráveis.

O programa verifica, de forma probabilística (por amostragem), se um conjunto de objetos (vetores ou matrizes) com duas operações dadas — adição entre elementos e multiplicação por escalar — satisfaz os 8 axiomas de espaços vetoriais.

## Estrutura do Projeto (Refatorado)

Este projeto foi refatorado para seguir o princípio da **Separação de Responsabilidades**. Em vez de um único ficheiro `main.py`, a lógica foi dividida em módulos:

  * `main.py`: Ponto de entrada da aplicação. Lida exclusivamente com a interface de linha de comando (`argparse`) e coordena os modos de execução.
  * `core.py`: Contém o "motor" do programa, a classe `VectorSpaceTester`, que é responsável pela lógica matemática de testar os 8 axiomas.
  * `utils.py`: Contém funções auxiliares para manipulação de dados, como parsing de JSON, criação de geradores de elementos e a definição dos *presets* de operações.
  * `ui_helpers.py`: Contém todas as funções que lidam com a interação com o utilizador no terminal (ex: `_prompt_int`, `_input_vector`, etc.).
  * `tests/`: Contém os testes automatizados (`unittest`) que verificam o funcionamento de todos os módulos.

-----

## Como Executar o Programa

Siga estes passos para instalar as dependências e correr a aplicação.

### 1\. Requisitos e Instalação

  * Python 3.9+
  * NumPy

Instale as dependências a partir do `requirements.txt`:

```powershell
pip install -r requirements.txt
```

### 2\. Execução Padrão (Não Interativo)

Para testar o espaço vetorial R^3 com as operações padrão (o caso que deve passar):

```powershell
python .\main.py --tipo vetor --dim 3 --operacoes padrao
```

Para testar matrizes 2x2:

```powershell
python .\main.py --tipo matriz --rows 2 --cols 2 --operacoes padrao
```

Para ver um teste a falhar intencionalmente (usando a "soma deslocada"):

```powershell
python .\main.py --tipo vetor --dim 2 --operacoes soma_deslocada --ensaios 50
```

### 3\. Execução (Modo de Entrada Manual)

Se preferir digitar os seus próprios vetores/matrizes, use o modo de entrada manual:

```powershell
python .\main.py --entrada-manual
```

O programa irá pedir-lhe o tipo, dimensões, o elemento zero e os elementos para o teste.

-----

## Como Verificar os Testes Automatizados 

Este projeto inclui uma suite de testes automatizados (`unittest`) para garantir que a lógica principal (em `core.py`, `utils.py`, etc.) está correta.

### 1\. Comando de Teste (Recomendado: Modo Silencioso)

Para correr todos os testes de forma limpa e rápida, use o comando `unittest` com a flag `-q` (quiet):

```powershell
python -m unittest discover tests -q
```

**Resultado esperado (Sucesso):**

```
................
----------------------------------------------------------------------
Ran 16 tests in 0.053s

OK
```

  * Cada `.` representa um teste que passou.
  * `OK` no final significa que a refatoração foi um sucesso e toda a lógica está correta.

### 2\. Comando de Teste (Modo Detalhado)

Se correr os testes sem a flag `-q`, verá muito mais texto, o que é **normal**.

```powershell
python -m unittest discover tests
```

**Entendendo a Saída Detalhada:**
A saída parecerá "confusa" porque os testes simulam o programa a ser executado, incluindo:

  * Simulações de utilizadores a digitar valores errados (`Forneça exatamente 3 valores...`).
  * Testes de falha intencional (`main.py: error: argument --operacoes: invalid choice: 'invalido'`).
  * Testes que verificam se os axiomas falham quando devem (`VEREDITO: ... NÃO é um Espaço Vetorial.`).

Tudo isto é **esperado** e faz parte da verificação. A linha final `OK` é o que confirma o sucesso.

-----

## Funcionalidades Avançadas

### Presets de operações

  * `padrao`: `u + v = u + v`, `k * u = k u` (Deve passar em todos os axiomas).
  * `soma_deslocada`: `u ⊕ v = u + v + 1` (Deve falhar A3, A4, A5, A6).
  * `escalar_estranho`: `k ⊗ u = (k+1) u` (Deve falhar A4, A6, A7, A8).

### Elementos e zero personalizados

Pode fornecer elementos específicos (em vez de aleatórios) via JSON:

  * `--elementos`: String JSON com uma lista de vetores/matrizes.
  * `--elementos-arquivo`: Caminho para um ficheiro JSON contendo a lista.
  * `--zero`: Elemento zero em formato JSON.

Exemplo:

```powershell
python .\main.py --tipo vetor --dim 2 --elementos "[[0,0],[1,2],[2,4]]" --zero "[0,0]"
```

### Operações customizadas (avançado)

Use lambdas em string com `--unsafe-eval` (cuidado: avalia código):

```powershell
python .\main.py --tipo vetor --dim 3 --unsafe-eval --add "lambda u,v: u+v" --sm "lambda u,k: k*u"
```

### Códigos de saída

  * `0` → todos os axiomas passaram.
  * `1` → algum axioma falhou.
  * `2` → erro de execução (ex.: JSON malformado).