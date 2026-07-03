---
name: moa-aggregation
description: Distilled heuristics for the MoA aggregator (auto-updated by `hermes moa evolve`)
metadata:
  hermes:
    auto_generated: moa-evolve
    updated: 2026-07-03T00:43:31Z
    turns_analyzed: 40
---

## Aggregation Heuristics

### When to Trust References
- Do not automatically follow the majority. If references disagree, independently solve the problem from scratch. (Turn 20: majority wrong; Turn 3, 7, 14, 19, 22, 25, 31)
- If a reference's reasoning contains obvious factual errors (e.g., missing a digit, incorrect prime factorization), discard that reference entirely. (Turn 8, 1, 33)
- A reference that fails to produce a final answer should be ignored; do not guess its intended answer. (Turn 11, 13, 16, 19, 29, 40)
- For tasks where all references are known to be unreliable (e.g., sum of digits of 3^50, sum of floor(100/k)), do not trust any reference; compute independently with a verified tool.

### Verification Habits
- For any counting task (letters, divisors, arrangements), manually recount or use a simple script; never trust a reference's count without verification. (Turn 3, 14, 22, 32)
- For arithmetic with large numbers (multiplication, exponentiation, sums), use an external tool (Python) and cross-check the result with a different method or a sanity check (e.g., approximate magnitude). (Turn 7, 13, 19, 23, 25, 31)
- When using a tool, do not hallucinate its output; actually run the code in a real environment. If the output seems off, re-run with a different approach. (Turn 7, 13, 19, 23, 31)
- For modular exponentiation, compute using Euler's theorem and repeated squaring; do not rely on pattern guessing or a single reference's cycle detection. Verify with `pow(base, exp, mod)`. (Turn 20, 35)
- For digit sums, the sum modulo 9 is a necessary but not sufficient check; compute the full number or use a reliable tool. (Turn 7, 25)
- For string slicing tasks, explicitly write the sorted string and index positions; double-check the slice boundaries. Before sorting, count the frequency of each character in the original string. (Turn 14, 32)

### Handling Disagreement
- If all references give different answers, solve the problem yourself step-by-step; do not pick among them. (Turn 3, 7, 14, 19, 22, 25, 31)
- If two references agree and one disagrees, still verify the agreed answer if the task is error-prone (e.g., modular arithmetic, large multiplication). (Turn 20, 35)
- When a reference's answer is an outlier and its reasoning is flawed, discount it heavily. (Turn 8, 15, 33)

### Task-Specific Pitfalls
- Trailing zeros in factorial: ensure each floor division (n/5, n/25, n/125, ...) is computed correctly; a single off-by-one error is common. (Turn 4, 38)
- Inclusion-exclusion: double-check each floor division and the final addition/subtraction. (Turn 15, 34)
- LCM/GCD: compute prime factorization carefully; verify LCM by checking that it is a multiple of both numbers. (Turn 18, 27)
- Probability: list all outcomes explicitly to avoid missing cases. (Turn 5, 36)
- Sequence recurrence: if using both iterative and closed-form, they must match; if not, recheck algebra. (Turn 17, 28)
- Counting letters in a phrase: write out each word and count the target letter manually; the correct count for the seashells text is 17. (Turn 3, 22)
- Sum of floor(100/k) for k=1..100: the correct answer is 482; any answer above 1000 is likely a misreading. Use the formula or a careful loop. (Turn 13, 31)
- Large integer multiplication: use Python's arbitrary precision and verify by checking the last few digits manually. (Turn 19, 23)
- String slicing after sorting: for 'mississippi', the sorted string is 'iiiimppssss' (4 i's, 1 m, 2 p's, 4 s's); slice [4:8] is 'mpps'. (Turn 14, 32)
- Base conversions: double-check the number of digits and the power for each position. (Turn 8, 33)
- CRT: verify the solution by testing all congruences; mistral is most reliable but still verify. (Turn 9, 24)
- Determinants: be careful with signs in cofactor expansion; gemma sometimes gets the sign wrong. (Turn 29)

## Reference model notes
- **llama-3.1-8b-instruct**: Frequently incorrect on arithmetic, counting, hex conversion, CRT, determinants, and Python la
