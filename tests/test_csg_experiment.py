from pathlib import Path

from csg_experiment import CSGExperiment, KnowledgeFact, SyntheticModel


def test_csg_beats_baseline_on_updated_facts():
    baseline = SyntheticModel({"current_president": "Joe Biden"})
    updated = SyntheticModel({"current_president": "Donald Trump"})

    experiment = CSGExperiment(
        questions=[
            {"question": "Who is the current president?", "expected": "Donald Trump"},
            {"question": "Who is the current president?", "expected": "Donald Trump"},
        ],
        baseline_model=baseline,
        csg_model=updated,
    )

    report = experiment.run()

    assert report["without_csg"] == 0.0
    assert report["with_csg"] == 1.0
    assert report["winner"] == "with_csg"


def test_experiment_generates_graph_file(tmp_path):
    baseline = SyntheticModel({"current_president": "Joe Biden", "language": "English"})
    updated = SyntheticModel({"current_president": "Donald Trump", "language": "English"})

    experiment = CSGExperiment(
        questions=[
            {"question": "Who is the current president?", "expected": "Donald Trump"},
            {"question": "What language is the system configured for?", "expected": "English"},
        ],
        baseline_model=baseline,
        csg_model=updated,
    )

    output_path = tmp_path / "experiment_chart.png"
    experiment.save_chart(output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
