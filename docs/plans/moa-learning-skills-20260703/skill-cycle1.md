---
name: moa-aggregation
description: Distilled heuristics for the MoA aggregator (auto-updated by `hermes moa evolve`)
metadata:
  hermes:
    auto_generated: moa-evolve
    updated: 2026-07-03T00:22:38Z
    turns_analyzed: 20
---

## Aggregation Heuristics

### When to Trust References
- Do not automatically follow the majority. If references disagree, independently solve the problem from scratch. (Turn 20: majority wrong)
- If a reference's reasoning contains obvious factual errors (e.g., missing a digit, incorrect prime factorization), discard that reference entirely. (Turn 8, 1)
- A reference that fails to produce a final answer should be ignored; do not guess its intended answer. (Turn 11, 13, 16, 19)

### Verification Habits
- For any counting task (letters, divisors, arrangements), manually recount or use a simple script; never trust a reference's count without verification. (Turn 3, 14)
- For arithmetic with large numbers (multiplication, exponentiation, sums), use an external tool (Python) and cross-check the result with a different method or a sanity check (e.g., approximate magnitude). (Turn 7, 13, 19)
- When using a tool, do not hallucinate its output; if possible, run the code in a real environment. If the output seems off, re-run with a different approach. (Turn 7, 13)
- For modular exponentiation, compute using Euler's theorem and repeated squaring; do not rely on pattern guessing or a single reference's cycle detection. (Turn 20)
- For digit sums, the sum modulo 9 is a necessary but not sufficient check; compute the full number or use a reliable tool. (Turn 7)
- For string slicing tasks, explicitly write the sorted string and index positions; double-check the slice boundaries. (Turn 14)

### Handling Disagreement
- If all references give different answers, solve the problem yourself step-by-step; do not pick among them. (Turn 3, 7, 14, 19)
- If two references agree and one disagrees, still verify the agreed answer if the task is error-prone (e.g., modular arithmetic, large multiplication). (Turn 20)
- When a reference's answer is an outlier and its reasoning is flawed, discount it heavily. (Turn 8, 15)

### Task-Specific Pitfalls
- Trailing zeros in factorial: ensure each floor division (n/5, n/25, n/125, ...) is computed correctly; a single off-by-one error is common. (Turn 4)
- Inclusion-exclusion: double-check each floor division and the final addition/subtraction. (Turn 15)
- LCM/GCD: compute prime factorization carefully; verify LCM by checking that it is a multiple of both numbers. (Turn 18)
- Probability: list all outcomes explicitly to avoid missing cases. (Turn 5)
- Sequence recurrence: if using both iterative and closed-form, they must match; if not, recheck algebra. (Turn 17)

## Reference model notes
- **llama-3.1-8b-instruct**: Frequently incorrect on arithmetic, counting, and modular arithmetic. Often provides flawed reasoning or no final answer. Low reliability; treat its answers as suspect, especially when it is the sole dissenter. It has occasional correct answers on straightforward tasks, but always verify.
- **ministral-8b-2512**: Moderate reliability. Strong on combinatorics, determinants, and Chinese Remainder Theorem. Prone to errors on digit sums, counting, floor sums, and string manipulation. Verify its answers on calculation-heavy tasks.
- **gemma-3-12b-it**: Moderate reliability. Strong on combinatorics, CRT, determinants, inclusion-exclusion. Prone to errors on counting, trailing zeros, digit sums, string slicing, large multiplication, and modular exponentiation. Verify on arithmetic and counting tasks.
