# Teste de Espaço Vetorial (CLI)

Este utilitário em Python testa, por amostragem, os 8 axiomas de espaço vetorial para conjuntos de vetores ou matrizes, com operações configuráveis. Você pode usá-lo totalmente via terminal (PowerShell no Windows).

## O que o código faz (visão detalhada)

O programa verifica, de forma probabilística (por amostragem), se um conjunto de objetos (vetores ou matrizes) com duas operações dadas — adição entre elementos e multiplicação por escalar — satisfaz os 8 axiomas de espaços vetoriais.

Contrato básico:
- Conjunto V: vetores R^n ou matrizes R^{m×n} (números reais com NumPy).
- Operações: adição `u + v` e multiplicação escalar `k * u` (ou variações definidas pelos presets ou lambdas).
- Elemento zero: vetor/matriz nulo do mesmo formato.
- Comparações numéricas usam tolerâncias `rtol` e `atol` (via `numpy.allclose`).

Os 8 axiomas testados (rótulos A1–A8):
1. A1 Comutatividade da adição: u + v = v + u
2. A2 Associatividade da adição: (u + v) + w = u + (v + w)
3. A3 Existência de zero: u + 0 = u
4. A4 Existência de inverso aditivo: u + (−u) = 0
5. A5 Distributividade escalar sobre vetores: k*(u+v) = k*u + k*v
6. A6 Distributividade escalar sobre escalares: (k+m)*u = k*u + m*u
7. A7 Associatividade de escalares: (k*m)*u = k*(m*u)
8. A8 Identidade escalar: 1*u = u

Como a verificação é por amostragem, você escolhe a quantidade de ensaios (`--ensaios`). Quanto maior, maior a confiança no veredito.

## Requisitos

- Python 3.9+
- Pacotes:
  - numpy

Instale as dependências:

```powershell
pip install -r requirements.txt
```

## Uso básico

Ajuda geral:

```powershell
python .\main.py --help
```

Testar o espaço vetorial padrão (adição normal, multiplicação escalar normal) em R^3 com 200 ensaios:

```powershell
python .\main.py --tipo vetor --dim 3 --ensaios 200 --operacoes padrao
```

Testar matrizes 2x2 com operações padrão:

```powershell
python .\main.py --tipo matriz --rows 2 --cols 2 --operacoes padrao
```

## Presets de operações

- `padrao`: u + v = u + v, k * u = k u (o padrão de espaços vetoriais).
- `soma_deslocada`: u ⊕ v = u + v + 1 (bias de uns); viola axiomas (útil para ver falhas).
- `escalar_estranho`: k ⊗ u = (k+1) u; também viola axiomas.

Exemplo (mostrar falha):

```powershell
python .\main.py --tipo vetor --dim 3 --operacoes soma_deslocada --ensaios 50
```

Definições formais dos presets e expectativas:
- padrao: adição usual e escalar usual.
  - Deve satisfazer todos os axiomas em R^n e em R^{m×n} (veredito esperado: PARECE ser um espaço vetorial).
- soma_deslocada: adição alterada para u ⊕ v = u + v + c, com c = matriz/vetor de 1s; escalar usual.
  - A1 e A2: PASSAM (permanece comutativa e associativa, pois o deslocamento c se compensa em ambas as ordens/agrupamentos).
  - A3 (zero), A4 (inverso), A5, A6: FALHAM (não existe um zero global compatível e a distributividade com escalares se quebra).
  - A7 (associatividade escalar) e A8 (identidade escalar): PASSAM (operador escalar é o usual).
- escalar_estranho: escalar alterado para k ⊗ u = (k+1)·u; adição usual.
  - A3 (zero): PASSA.
  - A4 (inverso): FALHA, pois o “inverso” calculado via (−1)⊗u = 0, logo u + 0 ≠ 0.
  - A5: PASSA (distribui sobre a adição). A6: FALHA ((k+m+1) vs (k+m+2)).
  - A7: FALHA ((k*m+1) vs (k+1)(m+1)). A8: FALHA (1⊗u = 2u ≠ u).

## Elementos e zero personalizados

Você pode fornecer uma lista de elementos (para amostragem) e/ou o elemento zero via JSON:

- `--elementos`: string JSON com uma lista de vetores/matrizes.
- `--elementos-arquivo`: caminho de um arquivo JSON contendo a lista.
- `--zero`: elemento zero em JSON.

Exemplos (vetores 2D):

```powershell
# elementos explícitos (lista de vetores)
python .\main.py --tipo vetor --dim 2 --elementos "[[0,0],[1,2],[2,4]]" --zero "[0,0]"
```

Exemplo (matrizes 2x2 via arquivo):

Arquivo `matrizes.json`:

```json
[[[1,0],[0,1]], [[0,0],[0,0]], [[2,3],[4,5]]]
```

Comando:

```powershell
python .\main.py --tipo matriz --rows 2 --cols 2 --elementos-arquivo .\matrizes.json
```

## Escolha imediata da operação

Se você não passar `--operacoes`, ao iniciar o programa ele já exibirá um menu para escolher:

```
== Seleção de Operações ==
  1) padrao         (u+v, k*u)
  2) soma_deslocada (u+v+1)
  3) escalar_estranho ((k+1)*u)
Opção [1/2/3] [1]:
```

Basta digitar 1, 2 ou 3 (Enter escolhe 1). Isso ocorre mesmo no modo não interativo sempre que você omite `--operacoes`.

## Operações customizadas (avançado)

Use lambdas em string com `--unsafe-eval` (cuidado: avalia código):

```powershell
# Adição padrão, escalar padrão (apenas como exemplo)
python .\main.py --tipo vetor --dim 3 --unsafe-eval --add "lambda u,v: u+v" --sm "lambda u,k: k*u"
```

As lambdas recebem:
- `add(u, v)` → retorna mesmo tipo de `u`.
- `sm(u, k)` → retorna mesmo tipo de `u`.

Você pode usar `np` (NumPy) dentro das lambdas.

## Entrada manual simplificada (apenas números)

Se você quer sempre digitar os números dos vetores/matrizes, sem precisar passar JSON ou entrar no modo interativo completo, use:

```powershell
python .\main.py --entrada-manual
```

Esse modo pergunta somente:
- Tipo (vetor/matriz) e dimensões (se não informados por flags),
- Elemento zero (opcional),
- Lista de elementos (n vetores/matrizes),
- Operações (menu instantâneo),
- Número de ensaios (se não informado).

Você também pode combinar flags para adiantar escolhas:

```powershell
# Vetor 3D, perguntando apenas zero, elementos e operações
python .\main.py --entrada-manual --tipo vetor --dim 3 --ensaios 50

# Matriz 2x3
python .\main.py --entrada-manual --tipo matriz --rows 2 --cols 3
```

## Tutoriais por operação (passo a passo)

### 1) Operação padrão (u+v e k*u)

Objetivo: validar que R^n e R^{m×n} formam espaços vetoriais com as operações usuais.

- Vetor R^3 (não interativo, aleatório):
```powershell
python .\main.py --tipo vetor --dim 3 --operacoes padrao --ensaios 200
```

- Vetor R^3 (entrada manual):
```powershell
python .\main.py --entrada-manual --tipo vetor --dim 3 --ensaios 50
# Digite 0 0 0 como zero, e seus vetores conforme solicitado.
```

- Matriz 2×2 (não interativo):
```powershell
python .\main.py --tipo matriz --rows 2 --cols 2 --operacoes padrao --ensaios 100
```

Resultado esperado: todos os axiomas PASSAM e veredito PARECE ser um espaço vetorial.

### 2) Soma deslocada (u ⊕ v = u + v + 1)

Objetivo: observar falhas de axiomas quando a adição é deslocada por uma constante c ≠ 0.

- Vetor R^3 (não interativo):
```powershell
python .\main.py --tipo vetor --dim 3 --operacoes soma_deslocada --ensaios 50
```

- Entrada manual (qualquer dimensão/shape):
```powershell
python .\main.py --entrada-manual --tipo vetor --dim 2 --ensaios 20
# Siga os prompts para zero e elementos; ao escolher operação, selecione 2) soma_deslocada.
```

Resultado típico: PASSAM A1, A2, A7, A8; FALHAM A3, A4, A5, A6. Veredito: NÃO é um espaço vetorial.

### 3) Escalar estranho (k ⊗ u = (k+1)·u)

Objetivo: observar falhas de axiomas quando a multiplicação por escalar é alterada.

- Vetor R^2 (não interativo):
```powershell
python .\main.py --tipo vetor --dim 2 --operacoes escalar_estranho --ensaios 50
```

- Matriz 3×2 (entrada manual):
```powershell
python .\main.py --entrada-manual --tipo matriz --rows 3 --cols 2 --ensaios 20
# Siga os prompts e escolha 3) escalar_estranho no menu.
```

Resultado típico: PASSAM A3, A5; FALHAM A4, A6, A7, A8. Veredito: NÃO é um espaço vetorial.

## Tolerâncias e reprodutibilidade

- `--rtol` e `--atol` controlam a comparação numérica (via `numpy.allclose`).
- `--seed` fixa a semente do gerador aleatório para reprodutibilidade.

## Códigos de saída

- `0` → todos os axiomas passaram.
- `1` → algum axioma falhou.
- `2` → erro de execução (ex.: shape inválido, JSON malformado).

## Exemplos rápidos

```powershell
# R^3 padrão, 100 ensaios
python .\main.py --tipo vetor --dim 3 --operacoes padrao --ensaios 100

# Matrizes 3x2 padrão
python .\main.py --tipo matriz --rows 3 --cols 2 --operacoes padrao

# Mostrar falhas com operações não lineares
python .\main.py --tipo vetor --dim 2 --operacoes escalar_estranho --ensaios 50
```

## Observações

- O método é probabilístico: elementos são amostrados ao acaso (ou da lista fornecida). Aumente `--ensaios` para maior confiança.
- Ao fornecer elementos, garanta que todos tenham o shape correspondente ao tipo escolhido.

## Solução de problemas (FAQ rápido)
- “Nada aparece para digitar os números”: se você rodar sem `--interativo` e sem `--elementos`, após escolher a operação o programa pergunta se quer digitar manualmente; responda `S`. Ou use `--entrada-manual` diretamente.
- “Erro de shape”: verifique `--dim` (vetor) ou `--rows/--cols` (matriz) e se os elementos digitados têm o mesmo tamanho.
- “Resultados variam”: aumente `--ensaios` e/ou fixe `--seed` para reprodutibilidade.
