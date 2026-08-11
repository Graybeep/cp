# Email draft — interpscience CFP format query (2026-08-09)

**To:** interpscience@gmail.com
**Why:** the CFP states "short (up to 5 pages) and long papers (up to 9 pages)" but is
silent on both references and appendices. XAI4Science states explicitly that both are
excluded; interpscience does not.

**PROMOTED TO BLOCKING, 2026-08-09.** The note below used to say the answer "does not
change the plan" at a +5-page overflow. That was true then and is false now: the leading
plan compiles to 8 pages with the figure consolidation — exactly the limit, nothing spare
— and `move_cost` rows C–E get there by moving §2 material *into an appendix*.
`logs/plan_compile.md` §4 compiles the reversal: **with the §2 material back in the body
the paper is 9 pp, not 8.** So a "no appendices" answer costs a page the plan does not
have. This is a worked example of a question whose value rose while it sat in the queue —
the overflow shrank underneath it.

**Not sent, and I cannot send it.** `mcp__claude_ai_Gmail__authenticate` returns "ask the
user to run `/mcp` and select claude.ai Gmail" — the OAuth flow only the account holder
can complete, so there is no in-session route and re-attempting the connector is not the
fix. Either authorise it via `/mcp` and I will send, or send from `ashwinsai102@gmail.com`
directly. **The draft still signs off `[name]`** — it needs a real name first.

---

**Subject:** Long paper track — do references and appendices count toward the 9-page limit?

Hello,

I'm preparing a submission for the Interpretability as a Science workshop and had two
formatting questions the call for papers doesn't cover.

1. Do references count toward the 9-page limit for long papers, or are they excluded as
   they are at the NeurIPS main conference?
2. Are appendices or supplementary material permitted, and if so, are they also excluded
   from the 9 pages?

Thank you, and thanks for organising the workshop.

Best regards,
[name]

---

## If a reply arrives, what it changes

| answer | consequence |
|:--|:--|
| references excluded, appendices allowed and excluded | the working assumption; the leading plan fits at 8 pp (ACL) / 9 pp is the interpscience limit, so it fits with a page in hand |
| references **counted** | roughly 0.8 further page to find — references are unlimited at ACL but not necessarily here |
| appendices **not** permitted | **costs a compiled page** — kills the §2 Methods move (`logs/move_cost.md` rows C–E) and the plan goes 8 pp → 9 pp (`logs/plan_compile.md` §4) |

The third row is the one that redirects work, and it is no longer a 0.23-page rounding
question: it is measured at a whole page, because the plan now sits on the limit rather
than five pages above it. Ask before the §2 move is executed, and before the merged
figure is drawn against an 8-page target that a "no" would invalidate.
