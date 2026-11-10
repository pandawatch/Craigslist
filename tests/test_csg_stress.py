from csg_demo import KnowledgeFact
from csg_stress_test import StressBenchmark, StressScenario, build_default_scenarios


def test_stress_benchmark_exercises_actual_graft_and_reports_regressions():
    report = StressBenchmark(build_default_scenarios()).run()

    assert report["evaluation_kind"] == "exact_fact_lookup_simulation"
    assert report["total_questions"] == 7
    assert report["baseline"]["correct"] == 5
    assert report["csg"]["correct"] == 6
    assert report["csg"]["regressions"] == 1
    assert report["csg"]["improvements"] == 2
    assert report["csg"]["accuracy"] == 6 / 7
    assert report["baseline"]["accuracy"] == 5 / 7


def test_low_confidence_conflict_preserves_baseline_value():
    scenario = StressScenario(
        name="weak update",
        baseline_knowledge={"security_policy": "require MFA"},
        adapter_facts=[
            KnowledgeFact(
                key="security_policy",
                statement="Legacy authentication is permitted.",
                value="allow legacy auth",
                confidence=0.4,
            )
        ],
        questions=[
            {"key": "security_policy", "question": "What is the security policy?", "expected": "require MFA"}
        ],
    )

    report = StressBenchmark([scenario]).run()

    assert report["baseline"]["accuracy"] == 1.0
    assert report["csg"]["accuracy"] == 1.0
    assert report["scenarios"][0]["conflicts"][0]["resolution"] == "core-wins"


def test_graft_confidence_boundary_across_101_values():
    for step in range(101):
        confidence = step / 100
        core = {"fact": "old"}
        adapter = KnowledgeFact("fact", "Updated fact", "new", confidence)
        system = StressBenchmark(
            [
                StressScenario(
                    name="threshold sweep",
                    baseline_knowledge=core,
                    adapter_facts=[adapter],
                    questions=[{"key": "fact", "question": "What is the fact?", "expected": "new"}],
                )
            ]
        )

        report = system.run()
        expected = "new" if confidence >= 0.8 else "old"
        actual = report["scenarios"][0]["csg_details"][0]["actual"]
        assert actual == expected


def test_stress_benchmark_generates_separate_chart(tmp_path):
    scenario = StressScenario(
        name="update",
        baseline_knowledge={"project_phase": "alpha"},
        adapter_facts=[
            KnowledgeFact(
                key="project_phase",
                statement="Project phase is beta.",
                value="beta",
                confidence=0.95,
            )
        ],
        questions=[{"key": "project_phase", "question": "What phase?", "expected": "beta"}],
    )

    chart_path = tmp_path / "validation_chart.png"
    result_path = StressBenchmark([scenario]).save_chart(chart_path)

    assert result_path == chart_path
    assert chart_path.exists()
    assert chart_path.stat().st_size > 0
