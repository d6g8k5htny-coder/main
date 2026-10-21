"""Offline routing consistency only; no live board, custody or status claims.

Proposed path: tests/test_active_board_routing.py. The canonical destination is
read from AGENTS.md, not hard-coded. Scope is the current simple inline-link
labels in RESEARCH_INDEX.md; this is not a general Markdown/HTML parser.
Historical links are untouched. Matching stale documents can still agree:
consistency does not establish the live board, access or authorisation.
"""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
ISSUE_PREFIX = "https://github.com/d6g8k5htny-coder/main/issues/"
LINK = re.compile(r"(?<!!)\[([^\]\n]+)\]\(([^)\n]*)\)")
CANONICAL = re.compile(r"Agent Message Board #([1-9][0-9]*)", re.I)
ACTIVE = re.compile(r"(?:active|current) Agent Message Board, main #([1-9][0-9]*)", re.I)


def visible_links(text):
    """Read simple inline links outside top-level backtick/tilde fences."""
    fence = None
    for number, line in enumerate(text.splitlines(), 1):
        token = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line)
        if fence is not None:
            if (token and token[1][0] == fence[0]
                    and len(token[1]) >= len(fence) and not token[2].strip()):
                fence = None
            continue
        if token:
            fence = token[1]
            continue
        for match in LINK.finditer(line):
            yield number, " ".join(match[1].split()), match[2]


def route_problems(agents, research_index):
    canonical = [(line, CANONICAL.fullmatch(label), target)
                 for line, label, target in visible_links(agents)
                 if CANONICAL.fullmatch(label)]
    if len(canonical) != 1:
        return ["AGENTS.md: expected exactly one canonical board link"]
    line, label, target = canonical[0]
    expected = ISSUE_PREFIX + label[1]
    if target != expected:
        return [f"AGENTS.md:{line}: canonical label/target mismatch"]
    problems = []
    count = 0
    for line, label, target in visible_links(research_index):
        active = ACTIVE.fullmatch(label)
        if not active:
            continue
        count += 1
        if active[1] != canonical[0][1][1] or target != expected:
            problems.append(f"docs/RESEARCH_INDEX.md:{line}: active board mismatch")
    if not count:
        problems.append("docs/RESEARCH_INDEX.md: no qualifying active board links")
    return problems


class RoutingContractTests(unittest.TestCase):
    def setUp(self):
        self.agents = self.canonical(475)
        self.good = self.active(475)

    @staticmethod
    def canonical(number):
        return f"[Agent Message Board #{number}]({ISSUE_PREFIX}{number})\n"

    @staticmethod
    def active(number, word="active"):
        return f"[{word} Agent Message Board, main #{number}]({ISSUE_PREFIX}{number})\n"

    def test_matching_active_and_current(self):
        self.assertEqual(route_problems(self.agents, self.good + self.active(475, "current")), [])

    def test_stale_active_board_is_detected(self):
        self.assertEqual(route_problems(self.agents, self.active(419)),
                         ["docs/RESEARCH_INDEX.md:1: active board mismatch"])

    def test_label_and_target_are_both_checked(self):
        for source in (self.good.replace("#475]", "#419]"),
                       self.good.replace("/475)", "/419)")):
            with self.subTest(source=source):
                self.assertEqual(len(route_problems(self.agents, source)), 1)

    def test_all_active_links_are_checked(self):
        self.assertEqual(route_problems(self.agents, self.good + self.active(419)),
                         ["docs/RESEARCH_INDEX.md:2: active board mismatch"])

    def test_historical_links_and_comment_anchors_remain_allowed(self):
        history = f"[#419]({ISSUE_PREFIX}419) [working guidance]({ISSUE_PREFIX}229#issuecomment-6038554880)\n"
        self.assertEqual(route_problems(self.agents, self.good + history), [])

    def test_fenced_examples_are_ignored(self):
        for opener, closer in (("```md", "```"), ("~~~~text", "~~~~")):
            with self.subTest(opener=opener):
                example = opener + "\n" + self.active(419) + closer + "\n"
                self.assertEqual(route_problems(self.agents, example + self.good), [])

    def test_shorter_or_different_fence_does_not_close_example(self):
        example = "````md\n```\n~~~\n" + self.active(419) + "````\n"
        self.assertEqual(route_problems(self.agents, example + self.good), [])

    def test_fenced_canonical_example_does_not_add_ambiguity(self):
        example = "```md\n" + self.canonical(419) + "```\n"
        self.assertEqual(route_problems(self.agents + example, self.good), [])

    def test_canonical_link_must_exist_and_be_unique(self):
        for agents in ("", self.agents + self.agents, self.agents + self.canonical(419)):
            with self.subTest(agents=agents):
                self.assertEqual(route_problems(agents, self.good),
                                 ["AGENTS.md: expected exactly one canonical board link"])

    def test_canonical_label_target_mismatch_is_refused(self):
        agents = self.agents.replace("/475)", "/419)")
        self.assertEqual(route_problems(agents, self.good),
                         ["AGENTS.md:1: canonical label/target mismatch"])

    def test_missing_active_label_is_not_vacuous_success(self):
        for index in ("", f"[#419]({ISSUE_PREFIX}419)", "```\n" + self.good + "```\n"):
            with self.subTest(index=index):
                self.assertEqual(route_problems(self.agents, index),
                                 ["docs/RESEARCH_INDEX.md: no qualifying active board links"])

    def test_wrong_destination_variants_are_refused(self):
        for target in (ISSUE_PREFIX + "475#issuecomment-1", ISSUE_PREFIX + "475?x=1",
                       ISSUE_PREFIX.replace("/main/", "/Math-/") + "475",
                       ISSUE_PREFIX.replace("github.com", "example.com") + "475",
                       ISSUE_PREFIX.replace("https:", "http:") + "475"):
            with self.subTest(target=target):
                index = f"[active Agent Message Board, main #475]({target})"
                self.assertEqual(len(route_problems(self.agents, index)), 1)

    def test_future_rollover_needs_no_hardcoded_board_update(self):
        self.assertEqual(route_problems(self.canonical(900), self.active(900)), [])
        self.assertEqual(len(route_problems(self.canonical(900), self.good)), 1)

    def test_case_and_whitespace_normalization(self):
        index = self.good.replace("active Agent Message Board", "CURRENT  agent message board")
        self.assertEqual(route_problems(self.agents, index), [])

    def test_image_is_not_an_active_navigation_link(self):
        self.assertEqual(route_problems(self.agents, "!" + self.good),
                         ["docs/RESEARCH_INDEX.md: no qualifying active board links"])


class RepositoryRoutingTests(unittest.TestCase):
    def test_research_index_matches_agent_entry(self):
        agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
        index = (ROOT / "docs/RESEARCH_INDEX.md").read_text(encoding="utf-8")
        self.assertEqual(route_problems(agents, index), [])


if __name__ == "__main__":
    unittest.main()
