    def test_fence_closes_only_with_same_character(self):
        example = "```md\n~~~\n" + self.active(419) + "```\n"
        self.assertEqual(route_problems(self.agents, example + self.good), [])

    def test_fence_with_info_string_does_not_close(self):
        example = "```md\n```text\n" + self.active(419) + "```\n"
        self.assertEqual(route_problems(self.agents, example + self.good), [])

    def test_four_space_indent_is_not_a_fence(self):
        index = "    ```\n" + self.active(419)
        self.assertEqual(route_problems(self.agents, index),
                         ["docs/RESEARCH_INDEX.md:2: active board mismatch"])

    def test_only_exact_active_labels_qualify(self):
        index = f"[see the active Agent Message Board, main #419 here]({ISSUE_PREFIX}419)\n"
        self.assertEqual(route_problems(self.agents, self.good + index), [])

    def test_only_exact_canonical_labels_qualify(self):
        agents = self.agents + f"[the Agent Message Board #419 archive]({ISSUE_PREFIX}419)\n"
        self.assertEqual(route_problems(agents, self.good), [])

    def test_canonical_label_is_case_insensitive(self):
        agents = self.agents.replace("Agent Message Board", "agent message board")
        self.assertEqual(route_problems(agents, self.good), [])

    def test_every_stale_active_link_is_reported(self):
        self.assertEqual(route_problems(self.agents, self.active(419) + self.active(307)),
                         ["docs/RESEARCH_INDEX.md:1: active board mismatch",
                          "docs/RESEARCH_INDEX.md:2: active board mismatch"])

