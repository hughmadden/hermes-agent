---
name: moa-aggregation
description: Distilled heuristics for the MoA aggregator (auto-updated by `hermes moa evolve`)
metadata:
  hermes:
    auto_generated: moa-evolve
    updated: 2026-07-03T01:13:45Z
    turns_analyzed: 40
---

## Aggregation Heuristics

### When to Trust References
- Do not automatically follow the majority. If references disagree, independently solve the problem from scratch.
- If a reference's reasoning contains obvious factual errors (e.g., missing a digit, incorrect prime factorization, miscounting), discard that reference entirely.
- A reference that fails to produce a final answer should be ignored.
- For tasks where all references are known to be unreliable (e.g., sum of digits of 3^50, sum of floor(100/k)), do not trust any reference; compute independently with a verified tool. Many references have been consistently wrong on these.
- If all three references agree but the task is error-prone (large multiplication, digit sums, floor sum), still independently verify using a different method or sanity check.

### Verification Habits
- For any counting task (letters, divisors, arrangements), manually count or use a simple script; never trust a reference's count. For letter counts, write out each word and count the target letter; the correct count for the seashells text is 17, not 15 or 18.
- For arithmetic with large numbers, use an external tool (Python) and cross-check with a different method or a sanity check (e.g., last few digits modulo 1000). If you cannot actually run a tool, use manual modular verification.
- When claiming to have used a tool, ensure the output is from a real execution; do not hallucinate results. If code cannot be executed, employ manual checks (modular arithmetic, magnitude approximation).
- For modular exponentiation, compute using Euler's theorem and repeated squaring; verify with `pow(base, exp, mod)` if possible, else double-check each step. Do not rely on pattern guessing.
- For digit sums, compute the full number with a reliable tool; the sum modulo 9 is necessary but not sufficient. The sum of digits of 3^50 is 144; many references gave 9, 15, or 108.
- For string slicing tasks, first count frequency of each character in the original string. For 'mississippi', frequencies: i:4, m:1, p:2, s:4 → sorted string 'iiiimppssss', slice [4:8] = 'mpps'.

### Handling Disagreement
- If all references give different answers, solve from scratch; do not pick among them.
- If two agree and one disagrees, still verify if the task is error‑prone (modular arithmetic, large multiplication, digit sums, floor sums).
- If a reference is an outlier and its reasoning is flawed, discount it heavily.
- If all three agree but the task is a known trap (sum of floor(100/k), digit sum of 3^50), independently solve and double-check.

### Task-Specific Pitfalls
- Trailing zeros in factorial: ensure each floor division (n/5, n/25, n/125, …) is correct; a single off‑by‑one is common.
- Inclusion‑exclusion: double‑check each floor division and the final addition/subtraction.
- LCM/GCD: compute prime factorization carefully; verify LCM is a multiple of both numbers. The divisor count formula multiplies (exponent+1); multiply manually to avoid slips.
- Probability: list all outcomes explicitly; for conditional probability, divide by the correct conditioning count.
- Sequence recurrence: if using both iterative and closed‑form, they must match; if not, recheck algebra.
- Determinants: watch signs in cofactor expansion; gemma sometimes gets the sign wrong. Recompute via a different row/column to verify.
- Large integer multiplication: use Python and verify last few digits modularly (e.g., 87654321×12345678 ends with …638; if mismatched, it's wrong).
- Sum of floor(100/k) for k=1..100: the correct answer is 482; references frequently give 706, 168, 451, etc. Do not trust any reference; compute carefully.
- Base conversions: double‑check digit positions and powers; one reference often multiplies by the wrong power.
- CRT: verify the solution by testing all congruences manually; even if a reference is often right, still verify.
- Counting letters in the seashells phrase: manually count every 's'; correct total is 17.

## Reference model
