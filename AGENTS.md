# AGENT COMMANDMENTS
> Universal coding standards, resident in every harness via this file (`AGENTS.md`).
> Version: 2.0.0
> The mechanically-checkable rules are enforced by tooling — `.pre-commit-config.yaml`
> + CI — and are not restated here. This file carries only what a linter cannot
> check: judgment and conduct. Project-specific context lives elsewhere.

## How to read this

Rules come in two kinds.

**Absolutes** are objective and enforced in pre-commit and CI: file and function
length, no `print`, no force-unwrap, no hardcoded secrets or literals, no bare
`catch`, no unused or commented-out code, secret scanning, and layer/vertical
boundaries. Violate one and the build fails — you do not have to hold them in
memory. They live in the repo's lint and CI configs.

**Judgment** is everything below. It requires interpretation, a linter cannot
enforce it, and it is the reason this file exists. Read it fully.

**Precedence.** When rules conflict: correctness and security > testability >
clarity > brevity and size limits.

**Decide once.** A judgment call, once made and noted, is settled for the session —
do not re-litigate it on a later turn. If a gate blocks you, see §3 "When a gate
fails". The answer is never to weaken the gate.

**Branch first.** Never edit on the default branch. Before your first edit, create
a short-lived branch named for the change. Every change reaches `main` through a
pull request: you open the PR with checks green and stop — a human merges. Branch
protection rejects direct pushes, so working on `main` only wastes the work.

---

## §1 — Code Quality (judgment)

**I. Less code is better.** Write the minimum that solves the problem — nothing
speculative, no abstraction for single-use code, no handling for states that
cannot occur. If you wrote 200 lines and it could be 50, rewrite it. The test:
would a senior engineer call this overcomplicated? When two solutions are equal,
take the shorter.

**II. Split for cohesion.** (The 300-line file and 40-line function limits are
tool-enforced.) When you must split, keep the extracted piece in the feature or
vertical it serves — not a global `Helpers/`, `Utils/`, or `Protocols/` bucket. A
coherent 320-line file beats two coupled 160-line files that reach into each
other; a component's props belong with the component. If no clean seam exists,
flag it rather than forcing a bad split.

**III. Name what carries meaning.** (Hardcoded literals, secrets, URLs, and ports
are tool-enforced.) The judgment is which numbers earn a name: domain constants,
limits, and thresholds do; trivial local arithmetic (`width / 2`, `index + 1`)
does not. Meaning, not magnitude.

**IV. Single responsibility.** A type that does two things is two types. Watch for
names containing "And/Manager/Handler/Helper/Utils", more than five injected
dependencies, or a type that both owns data and renders it. Prevent the god
object — refactoring one is always harder than avoiding it.

**V. Interface-first, at the seams you test.** Anything you swap for a test double —
services, external dependencies, I/O — goes through an interface. But abstract
only where you actually test or replace: don't wrap the standard library or add an
interface with one permanent implementation. Over-abstraction fights I.

**VI. Handle errors meaningfully.** (Bare `catch`/`except` is tool-enforced.) Catch
specific types; a handler either logs with context or rethrows. A `catch` that
hides a failure converts a diagnosable bug into an invisible one.

**VII. No deferred crashes.** (Force-unwrap is tool-enforced.) Guard, bind, or
coalesce with a meaningful fallback. A deliberate fail-fast is different from a
convenience crash: when a violated invariant means state is already corrupt,
halting with a clear message beats continuing and mangling user data.
`precondition` survives release and is the tool for that; `assert` compiles out;
`fatalError` is for impossible-by-construction states only.

**VIII. New logic ships with tests.** Reframe the task as a verifiable goal: "add
validation" → write tests for the invalid inputs, then make them pass; "fix the
bug" → write a failing test that reproduces it, then make it pass. Name tests by
behaviour, not implementation. If a unit can't be tested without a real database,
network, or UI, the design is wrong (see V) — fix the design first.

**IX. Concurrency is explicit and contained.** (The compiler enforces what it can.)
No benign data race: shared mutable state reached from two contexts is
synchronised. UI state mutates only on the main thread/actor. Every async path has
defined cancellation and failure behaviour.

**X. Validate untrusted input at the boundary.** External data — user input,
network, files, IPC, scraped content — is untrusted until validated, once, at the
edge. Never build SQL, a shell command, a script, or markup by interpolating
untrusted values: parameterise, pass argument arrays, escape.

**XI. Resilient I/O.** Every network, IPC, or process call has a timeout. Any
external mutation is idempotent or guarded against double-execution. Retries use
bounded backoff, not a tight loop.

**XII. Justify every dependency.** A dependency is a permanent liability. Prefer
the standard library for anything you could write in a few lines. Before adding a
package, confirm it exists and is the genuine, maintained one — plausible names
are often typosquats or hallucinations.

**XIII. Tests are deterministic and hermetic.** No dependence on the wall clock,
real randomness, the network, or the live filesystem — inject the clock, seed the
randomness, stub the I/O. Real resources only in tests explicitly marked
integration.

**XIV. Make illegal states unrepresentable.** Push invariants into the type system
so the bad state cannot be constructed: an enum over a boolean-plus-meaning, a
non-optional field over "we promise it's set", a value type over a bare `String`
for a domain concept. The check you delete is the check that can't be skipped.

**XV. Release what you acquire.** Files, connections, observers, timers, tasks —
acquire and release in the same scope, or tie release to the owner's lifetime.
Break retain cycles (weak capture in escaping closures; tear down observers and
timers). "Works for an hour then slows down" is almost always a leak.

**XVI. Atomic writes, forward-only migrations.** Writes that must all-succeed-or-
all-fail run in one transaction. Migrations are forward-only, idempotent, and
tested against real data shapes; a shipped migration is never edited — you add the
next one.

**XVII. User-facing errors are not log messages.** What a user sees is actionable
and free of internals — never a stack trace, exception string, SQL, or path.
Diagnostics go to the log; the user gets something they can act on.

**XVIII. Secrets are a lifecycle.** (Secret scanning is tool-enforced.) The rest:
tokens scoped to least privilege, nothing secret or personal in a URL or query
string, `.env` files gitignored.

**XIX. Degrade gracefully; fail safe.** When a dependency is down, degrade
predictably rather than crash the flow. On any failure path, fail in the safe
direction: deny rather than grant, preserve rather than drop, stop rather than
guess.

**XX. Time is UTC, zoned at the edges, measured monotonically.** Persist and
compute in UTC; convert to local only where a human reads it, explicitly. Measure
durations with a monotonic clock — wall-clock subtraction goes negative on NTP
sync and at DST transitions.

**XXI. Document the public interface.** Exported functions and types document
their contract — preconditions, failure behaviour, ownership, units — not a
restatement of the signature. Private helpers get nothing; I still governs.

**XXII. Accessible by default (UI only).** Where there is a UI: every control
reachable and operable by keyboard, labelled for assistive technology, contrast
meeting the standard (via the token system), motion and text honouring
reduced-motion and dynamic-type. Designed in, it's cheap; retrofitted, a rewrite.

---

## §2 — Architecture

**Layered call path.** Always `Presentation → Service/ViewModel → Repository/Provider → Data`.
UI never holds a database connection or calls an API directly — that boundary is
what makes each layer testable and replaceable.

**Concerns by layer.** Presentation renders state and handles input; business logic
orchestrates, validates, transforms; data stores and retrieves; network fetches
and deserialises. None reaches into another's job.

**Config outside code.** Keys, URLs, flags, timeouts, ports live in environment or
config, never in source. The codebase deploys to any environment by changing
configuration, not code.

**Dependencies flow inward.** Higher-level modules depend on lower-level
abstractions, never the reverse. Circular dependencies are a design error —
resolve by extracting a shared abstraction, not with a late import.

**Organise by vertical, not by technical type.** Group code by what it does — one
feature's components, logic, types, and helpers live together (`src/billing/`), not
scattered across global `components/`, `hooks/`, `types/`, `utils/`. Grouping by
type splits code that changes together and produces un-navigable trees — the cost
is paid on every task, agents most of all. This is orthogonal to the layer rule:
you still separate presentation/logic/data, but *within* a vertical. Each vertical
declares a public interface and keeps the rest private; the boundary is
tool-enforced (a violation is a build error). Shared code is its own vertical, not
a dumping ground. Monorepo / package-per-vertical only when scale or multiple
owners justify it.

---

## §3 — Agent Conduct

How you behave while producing code. These address failure modes specific to
agents, and a linter cannot catch most of them.

**Think before coding.** State assumptions explicitly; if uncertain, ask. If
multiple interpretations exist, surface them — don't pick silently. If a simpler
approach exists, say so. If something is unclear, stop and name it.

**Verify, never invent.** Don't use an API, method, config key, or package you
haven't confirmed exists — check the actual type, signature, or docs first.

**No stub passed off as done.** Never return placeholder or fake data, leave a body
as "implement later", or call a half-built function finished. If you can't complete
it, say so and stop.

**Never weaken a check to pass it.** Fix the code, never disable the gate: no
`type: ignore`, `@ts-ignore`, `eslint-disable`, `--no-verify`, widening a type to
escape the checker, loosening an assertion, or skipping a test. Suppressing the
signal is worse than the failure it hides.

**Solve the general case, not the test.** Satisfy the requirement, not the test's
specific inputs. Hardcoding expected outputs is a defect that ships, not a passing
test.

**Stay in scope.** One task per change. No drive-by refactors, no reformatting
files you weren't asked to touch. Every changed line should trace to the request.

**Read before you write.** Read the surrounding module and match its patterns,
naming, and style — even if you'd do it differently. Code that ignores local
convention is wrong in context.

**Don't destroy what you don't understand.** Preserve existing behaviour unless
changing it is the task. Before deleting code, confirm it's actually unused. If you
notice unrelated dead code, mention it — don't delete it.

**Verify before reporting done.** "Done" means you ran the build, tests, and linter
and they passed — not that you believe they would. For a multi-step task, state the
plan as steps, each with a verification check. Report what you ran.

**When a gate fails.** A hard gate is to be obeyed, not argued with. Read the error
and fix the cause; if it still fails, try at most twice more, then stop and report
the violation, what you tried, and why you're stuck — hand back to a human. Never
loop indefinitely, never take a forbidden exit to escape a corner. The way out is
escalation, not suppression.

---

## Changelog

**2.0.0** — MAJOR (restructure)
- relocated all mechanically-checkable rules to tooling (`.pre-commit-config.yaml`
  + CI); this file now carries only judgment and conduct, and is the resident core
  loaded by every harness via `AGENTS.md`
- renamed `AGENT_COMMANDMENTS.md` → `AGENTS.md` for cross-harness auto-loading
  (Claude Code via a CLAUDE.md import; Antigravity, OpenCode, Zed natively)
- added the branch-first / PR rule to the preamble
- folded in the behavioural guidance from the prior global `CLAUDE.md`
  (think-before-coding, simplicity, surgical changes, goal-driven execution) and
  its sharper phrasings (the 200→50 test, test-first reframing, "every changed line
  traces to the request", "mention dead code, don't delete it")
- §1 compressed to judgment essence; the §4 NEVER block, the §6 enforcement
  tables, and the per-language tables were removed — they now live in the configs

Prior history 1.0.0–1.3.0 is preserved in the full `AGENT_COMMANDMENTS` reference.


## Project context (Esper)
Project: mygen-playground · Stack: Python

### Esper skill routing
Before starting any non-trivial task, call pil_route_skill("<describe your task>").
Do not load skills upfront. Load skill content only via pil_get_skill("<name>")
for skills you decide to use. Call pil_skill_used("<name>") after use.

### Esper task queue
For agent work sessions, call pil_get_tasks_by_tag to get the current work queue.
Prefer tasks with file references and error context over bare-title tasks.

### Esper MCP server
Available at localhost:4242. Health check: curl localhost:4242/health
