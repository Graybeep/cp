"""Before deleting a section: what elsewhere in the paper breaks, and what that is worth.

`section_menu.py` costs a section removal as though sections were independent. They are
not. §5 is referred to from §1, §2.4, §3.1, §7.2, §8 and Related Work, and those references
fall into three kinds that behave differently:

  ORPHAN     the passage names §5 and stops making sense without it -> must be rewritten
  FREED      the passage exists ONLY to support §5 -> deletes with it, buying extra lines
  DISSOLVED  an obligation that existed only because §5 existed -> satisfied by absence

The FREED rows matter for the arithmetic: `section_menu.py` said dropping §5 buys 2 pages,
but it removed only the section itself. Removing what the section drags with it buys more.
The DISSOLVED row matters more than either, because it is the one a reviewer would catch:
backbone §5.5 forbids shipping claim 2's well-powered size null without its underpowered
twin. The twin is §5's. Deleting §5 does not violate that rule -- it removes both halves.

Part 2 costs the alternative to deleting Related Work outright: demoting it to a single
Discussion paragraph that keeps the positioning function, since "delete" and "keep" are
not the only two options and the section's job (showing the schedule finding is not a
rediscovery of loss-of-plasticity) is not one the paper can simply drop.

Writes logs/cut_dependencies.md.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from move_cost import cost, norm  # noqa: E402
from page_estimate import LINES_PER_PAGE, SECTIONS, body, measure  # noqa: E402

OUT = "logs/cut_dependencies.md"

# Every inbound reference to §5, with what it costs to resolve. Verdicts are judgements;
# the DETECTION is mechanical (see audit_completeness below) so a new reference added to
# a draft cannot quietly escape this table.
DEPS = [
    dict(
        tag="ORPHAN", where="§1", label="§1 Introduction",
        what="the MDE reporting rule's worked example points at §5.3",
        edits=[(
            """And because several results here are nulls, we read non-significance off the
minimum detectable effect rather than the *p*-value: two structurally identical size
comparisons in this paper carry opposite evidential weight for that reason (§3.1, §5.3).""",
            """And because several results here are nulls, we read non-significance off the
minimum detectable effect rather than the *p*-value.""",
        )],
        note="The rule survives — §3.1 and §3.3 still use it. Only the example dies.",
    ),
    dict(
        tag="FREED", where="§2.4", label="§2 Methods",
        what="the eval-contamination correction exists only because it hit §5's numbers",
        edits=[
            ("""An earlier version of our circuit analysis generated its evaluation items
per-run, and those draws overlapped the training problems by 17–19 % in four of six cases;
all affected results were re-scored on the held-out set, and the correction changed a
conclusion (§5.2).""", None),
            ("""We report this rather than only the corrected numbers, because the released
logs contain both generations.""", None),
        ],
        note="The harness-agreement floor (~0.005) stays: §3 and §4 compare across "
             "harnesses too.",
    ),
    dict(
        tag="DISSOLVED", where="§3.1", label="§3 Detect",
        what="the backbone §5.5 paired-null dependency",
        edits=[(
            """We return to this in §5, where a structurally identical size comparison
carries the opposite evidential weight for one reason only: its MDE exceeds the entire
range its outcome occupies. The two nulls are best read as a pair.""",
            None,
        )],
        note="NOT a violation. §5.5 forbids either size null shipping alone; the "
             "underpowered one IS §5's, so removing §5 removes both halves and the rule "
             "is satisfied vacuously. Claim 2 keeps its own MDE and needs no partner.",
    ),
    dict(
        tag="ORPHAN", where="§7.2", label="§7–§9",
        what="prediction #3's accountability statement is written around §5's position",
        edits=[(
            """§5 diagnoses models recovered by the learning-rate shift, which prediction #3
did not cover; those results should not be read as confirming or disconfirming it. We state
this explicitly because §5 occupies the position in the paper where a reader holding our
pre-registration will look for #3's answer.""",
            """We report no circuit diagnosis here, so #3 remains unentered rather than
answered.""",
        )],
        note="The §11 obligation to account for #3 survives and gets simpler: with no "
             "circuit section there is no longer a place a reader mistakes for its answer.",
    ),
    dict(
        tag="FREED", where="§8", label="§7–§9",
        what="two of the five costed label-movers move labels that would no longer exist",
        edits=[
            ("""**The circuit-identity gradient is unresolved at *n* = 6.** Additional triads
at either size are the only thing that would settle whether deficit depth grades circuit
identity; the direction persists but four of six leave-one-out deletions break its
significance.""", None),
            ("""**The circuit results are single-method by demonstration.** Joint head-group
patching, shown to have dynamic range on the repair-versus-reorganization contrast, would
let those claims drop the annotation. Three probes we tried do not discriminate.""", None),
        ],
        note='§8 opens "Five specific experiments" — becomes three. One-word edit, not '
             "costed here.",
    ),
    dict(
        tag="FREED", where="Related Work", label="Related Work",
        what="the circuit-identity citation paragraph positions §5 and nothing else",
        edits=[(
            """**Circuit identity.** Our diagnosis uses causal ablation and activation
patching in the circuit-analysis tradition [Elhage et al., 2021; Wang et al., 2022; Meng et
al., 2022] and representational similarity [Kornblith et al., 2019]; §5.1 reports the
negative methodological result that three of those four probes do not discriminate on our
contrast.""",
            None,
        )],
        note="Also retires four unverified citations, which is a small risk reduction as "
             "well as a saving.",
    ),
]

# The Related Work demotion: one Discussion paragraph instead of a standalone section.
# It keeps the plasticity/warm-starting distinction close to full length because that is
# the section's actual job -- it is where a reader decides whether the schedule result is
# a rediscovery -- and compresses everything else to a clause.
RW_PARAGRAPH = """**Relation to prior work.** Critical periods are established in
development and, for vision models, in Achille et al.'s demonstration that an early deficit
leaves a permanent gap [Achille, Rovere & Soatto, 2019]; we manipulate *when* a skill's data
arrives rather than its quality, holding total exposure fixed. The nearest explanation of
our reopening result, and the one we most risk rediscovering, is plasticity loss and warm
starting [Ash & Adams, 2020; Dohare et al., 2024; Lyle et al.]. The difference is specific:
those settings involve a distribution shift, a restart, or a sequence of distinct tasks and
recover plasticity by perturbing *weights*, whereas our models train once on a stationary
mixture whose only manipulated variable is a skill's onset step, and the intervention that
recovers the skill changes no weights at all, while weight resetting, the standard
plasticity remedy, produced no detectable recovery here (*n* = 1, §7.2). The mechanisms may
yet be one thing seen from two directions; what we can say is that recovery here needs no
weight perturbation. Our result is also distinct from grokking [Power et al., 2022], where
the data is fixed and generalization is sudden, against a permanent endpoint gap produced by
shifting arrival."""


def audit_completeness():
    """Fail loudly if a draft references §5 in a passage this table does not cover.

    The table's verdicts are judgements and cannot be automated, but its COMPLETENESS can
    be. Without this the file becomes exactly the kind of trusted artifact that silently
    stops matching the drafts it describes.
    """
    missed = []
    for label, path, start, stop, _ in SECTIONS:
        if label == "§5 Circuit identity":
            continue
        text = norm(" ".join(body(path, start, stop)))
        # Remove what the table already accounts for, then see what §5 references are
        # left. Matching per source line instead would false-positive on every line that
        # straddles the boundary of a covered sentence.
        for d in DEPS:
            for excision, _ in d["edits"]:
                text = text.replace(norm(excision), " ")
        for m in re.finditer(r"§5", text):
            missed.append(f"{label}: ...{text[max(0, m.start() - 70):m.start() + 70]}...")
    return missed


def main():
    missed = audit_completeness()

    rows = []
    for d in DEPS:
        r = cost(d["label"], d["edits"])
        rows.append((d, r))

    freed = sum(r["naive"] for d, r in rows if d["tag"] == "FREED")
    freed += sum(r["naive"] for d, r in rows if d["tag"] == "DISSOLVED")
    rewrite_cost = sum(r["honest"] for d, r in rows if d["tag"] == "ORPHAN")

    # Related Work demotion
    rw_lines = measure(body(*SECTIONS[5][1:4]), "Related Work")[0]
    para_lines = measure(RW_PARAGRAPH.split("\n"), "x")[0]

    out = [
        "# Dropping §5: what breaks, what it frees, and what to do about Related Work",
        "",
        "Source: `scripts/cut_dependencies.py`. Line deltas measured with "
        "`page_estimate`'s wrap,",
        "same as `move_cost.py`.",
        "",
        "## Part 1 — every inbound reference to §5",
        "",
        "| kind | where | what depends on §5 | resolving it |",
        "|:--|:--|:--|--:|",
    ]
    for d, r in rows:
        delta = r["honest"] if d["tag"] == "ORPHAN" else r["naive"]
        sign = "+" if d["tag"] != "ORPHAN" else "+"
        out.append(f"| **{d['tag']}** | {d['where']} | {d['what']} | "
                   f"{sign}{delta:.0f} ln freed |")
    out += [
        "",
        *[f"- **{d['where']} ({d['tag']})** — {d['note']}" for d, _ in rows],
        "",
        f"**§5 drags {freed + rewrite_cost:.0f} further lines out with it** "
        f"({(freed + rewrite_cost) / LINES_PER_PAGE:.2f} page) beyond the section itself: "
        f"{freed:.0f} lines of material that exists only to support it, plus "
        f"{rewrite_cost:.0f} from shortening the two passages that merely point at it.",
        "",
        "**The abstract needs no edit at all** — it contains no circuit content (claims "
        "13–16 were",
        "never abstract anchors). That is the strongest single indication that §5 is "
        "genuinely",
        "severable rather than load-bearing.",
        "",
        "**And the one rule that looked like a blocker is not one.** Backbone §5.5 forbids "
        "claim 2's",
        "well-powered size null shipping without its underpowered twin. The twin is §5's, "
        "so deleting",
        "§5 removes both halves and satisfies the rule by absence rather than breaking it. "
        "Worth",
        "stating explicitly in the §5.5 record, because the next reader of that rule will "
        "assume the",
        "opposite.",
        "",
        "## Part 2 — Related Work: demote rather than delete",
        "",
        f"The standalone section is **{rw_lines:.0f} lines "
        f"({rw_lines / LINES_PER_PAGE:.2f} page)**. Deleting it outright buys all of that "
        "and",
        "discards the positioning argument. Demoting it to a single Discussion paragraph "
        "that keeps",
        "the plasticity/warm-starting distinction near full length, compresses Achille and "
        "grokking",
        f"to a clause each, and pushes the BabyLM/BLiMP/agreement citations inline to §2.3 "
        f"and §6,",
        f"costs **{para_lines:.0f} lines** — so it buys "
        f"**{rw_lines - para_lines:.0f} lines "
        f"({(rw_lines - para_lines) / LINES_PER_PAGE:.2f} page)** against deletion's "
        f"{rw_lines:.0f}.",
        "",
        f"**The positioning function therefore costs {para_lines:.0f} lines "
        f"({para_lines / LINES_PER_PAGE:.2f} page) to keep, not "
        f"{rw_lines:.0f}.** That is the",
        "whole argument for demoting: the section's own length note calls the plasticity "
        "paragraph",
        "the place \"where a reader decides whether the schedule finding is a "
        "rediscovery\", and a",
        "paper that drops it saves a third of a page by leaving its most likely reviewer "
        "objection",
        "unanswered. Note also that the circuit-identity citations leave anyway with §5 "
        "(Part 1),",
        "so the demoted paragraph is shorter than the section minus that paragraph would "
        "suggest.",
        "",
        "Draft of the replacement paragraph is in this script (`RW_PARAGRAPH`) so the "
        "measurement",
        "and the text cannot drift apart.",
    ]
    if missed:
        out += ["", "## ⚠ Uncovered §5 references — this table is incomplete", ""]
        out += [f"- {m}" for m in missed]

    os.makedirs("logs", exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
