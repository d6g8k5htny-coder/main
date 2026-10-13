import copy, sys, unittest
sys.path.insert(0, '<SCRATCH>/wt250/reviews/register_execution_readiness_20260930')
import test_witness_states as T

class Nested(T.WitnessCLI):
    def test_nested_integer_boolean(self):
        for stage in ("open", "reviewed"):
            for val in (1, 0):
                g = T.graph_fixture(stage)
                g["nodes"]["math.c6-component.palm-proof"]["review_basis"][0]["source_exposed"] = val
                run, result = self.run_graph(g)
                same = run.stdout == (T.PACKET / "RESULTS_INSTALLED.json").read_bytes()
                print(f"\nstage={stage} source_exposed={val!r}: exit={run.returncode} passed={result['passed']} "
                      f"stdout==pinned RESULTS_INSTALLED={same} failed_checks={[k for k,v in result['checks'].items() if not v]}")

if __name__ == "__main__":
    unittest.main(argv=["x", "Nested"], verbosity=0)
