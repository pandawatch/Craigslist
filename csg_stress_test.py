from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from csg_demo import ContinuousSynapticGrafting, KnowledgeAdapter, KnowledgeFact, StaticCore


@dataclass(frozen=True)
class StressScenario:
    name: str
    baseline_knowledge: Dict[str, str]
    adapter_facts: Sequence[KnowledgeFact]
    questions: Sequence[Dict[str, str]]


class StressBenchmark:
    def __init__(self, scenarios: Iterable[StressScenario]):
        self.scenarios = list(scenarios)

    @staticmethod
    def score_knowledge(knowledge: Dict[str, str], questions: Sequence[Dict[str, str]]) -> Dict[str, Any]:
        correct = 0
        details: List[Dict[str, Any]] = []
        for item in questions:
            key = str(item["key"])
            expected = str(item["expected"])
            actual = knowledge.get(key, "unknown")
            is_correct = actual.lower() == expected.lower()
            if is_correct:
                correct += 1
            details.append(
                {
                    "key": key,
                    "question": str(item["question"]),
                    "expected": expected,
                    "actual": actual,
                    "correct": is_correct,
                }
            )
        total = len(questions)
        return {
            "accuracy": correct / total if total else 0.0,
            "correct": correct,
            "total": total,
            "details": details,
        }

    def run(self) -> Dict[str, Any]:
        scenario_reports: List[Dict[str, Any]] = []
        baseline_correct = 0
        csg_correct = 0
        total_questions = 0
        improvements = 0
        regressions = 0

        for scenario in self.scenarios:
            baseline = dict(scenario.baseline_knowledge)
            baseline_result = self.score_knowledge(baseline, scenario.questions)

            grafted_core = StaticCore(scenario.baseline_knowledge)
            system = ContinuousSynapticGrafting(
                grafted_core,
                KnowledgeAdapter.from_fact_stream(scenario.adapter_facts),
            )
            system.graft()
            csg_result = self.score_knowledge(grafted_core, scenario.questions)

            for before, after in zip(baseline_result["details"], csg_result["details"]):
                if not before["correct"] and after["correct"]:
                    improvements += 1
                elif before["correct"] and not after["correct"]:
                    regressions += 1

            baseline_correct += baseline_result["correct"]
            csg_correct += csg_result["correct"]
            total_questions += baseline_result["total"]
            winner = (
                "with_csg"
                if csg_result["accuracy"] > baseline_result["accuracy"]
                else "without_csg"
                if baseline_result["accuracy"] > csg_result["accuracy"]
                else "tie"
            )
            scenario_reports.append(
                {
                    "name": scenario.name,
                    "without_csg": baseline_result["accuracy"],
                    "with_csg": csg_result["accuracy"],
                    "winner": winner,
                    "conflicts": system.conflicts,
                    "baseline_details": baseline_result["details"],
                    "csg_details": csg_result["details"],
                }
            )

        baseline_accuracy = baseline_correct / total_questions if total_questions else 0.0
        csg_accuracy = csg_correct / total_questions if total_questions else 0.0
        overall_winner = (
            "with_csg"
            if csg_accuracy > baseline_accuracy
            else "without_csg"
            if baseline_accuracy > csg_accuracy
            else "tie"
        )

        return {
            "evaluation_kind": "exact_fact_lookup_simulation",
            "limitations": [
                "This evaluates key-value fact updates, not a language model or trained LoRA adapter.",
                "Scenarios are hand-authored and results do not establish general model improvement.",
                "Accuracy is descriptive only; this benchmark does not estimate statistical significance.",
            ],
            "total_questions": total_questions,
            "baseline": {
                "correct": baseline_correct,
                "total": total_questions,
                "accuracy": baseline_accuracy,
            },
            "csg": {
                "correct": csg_correct,
                "total": total_questions,
                "accuracy": csg_accuracy,
                "improvements": improvements,
                "regressions": regressions,
            },
            "overall_without_csg": baseline_accuracy,
            "overall_with_csg": csg_accuracy,
            "overall_winner": overall_winner,
            "scenarios": scenario_reports,
            "scenario_results": scenario_reports,
        }

    def save_chart(self, output_path: str | Path) -> Path:
        try:
            import matplotlib.pyplot as plt
        except ModuleNotFoundError:  # pragma: no cover
            import subprocess
            import sys

            subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
            import matplotlib.pyplot as plt

        report = self.run()
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        labels = [entry["name"] for entry in report["scenarios"]]
        without_values = [entry["without_csg"] for entry in report["scenarios"]]
        with_values = [entry["with_csg"] for entry in report["scenarios"]]

        x = range(len(labels))
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar([i - 0.2 for i in x], without_values, width=0.35, label="Without CSG", color="#b8c2cc")
        ax.bar([i + 0.2 for i in x], with_values, width=0.35, label="With CSG", color="#4c78a8")
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Accuracy")
        ax.set_title("Stress Test: Baseline vs CSG")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend()

        for i, value in enumerate(without_values):
            ax.text(i - 0.2, value + 0.03, f"{value:.0%}", ha="center", va="bottom", fontsize=8)
        for i, value in enumerate(with_values):
            ax.text(i + 0.2, value + 0.03, f"{value:.0%}", ha="center", va="bottom", fontsize=8)

        fig.tight_layout()
        fig.savefig(output_file, dpi=150)
        plt.close(fig)
        return output_file


def build_default_scenarios() -> List[StressScenario]:
    return [
        StressScenario(
            name="Verified Update",
            baseline_knowledge={"current_president": "Old value", "chief_of_staff": "Mia Lopez"},
            adapter_facts=[
                KnowledgeFact("current_president", "Current president changed to New value", "New value", 0.95)
            ],
            questions=[
                {"key": "current_president", "question": "Who is the current president?", "expected": "New value"},
                {"key": "chief_of_staff", "question": "Who is the chief of staff?", "expected": "Mia Lopez"},
            ],
        ),
        StressScenario(
            name="Weak Conflicting Update",
            baseline_knowledge={"security_policy": "require MFA", "compliance_status": "approved"},
            adapter_facts=[
                KnowledgeFact("security_policy", "Legacy authentication allowed", "allow legacy auth", 0.4)
            ],
            questions=[
                {"key": "security_policy", "question": "What is the security policy?", "expected": "require MFA"},
                {"key": "compliance_status", "question": "What is the compliance status?", "expected": "approved"},
            ],
        ),
        StressScenario(
            name="Novel Fact",
            baseline_knowledge={"organization_type": "research lab"},
            adapter_facts=[
                KnowledgeFact("new_product", "New product launch is Atlas", "Atlas", 0.95)
            ],
            questions=[
                {"key": "new_product", "question": "What is the new product?", "expected": "Atlas"},
                {"key": "organization_type", "question": "What type of organization?", "expected": "research lab"},
            ],
        ),
        StressScenario(
            name="Bad Confident Update",
            baseline_knowledge={"stable_version": "4.2"},
            adapter_facts=[
                KnowledgeFact("stable_version", "Version changed to 9.9", "9.9", 0.97)
            ],
            questions=[
                {"key": "stable_version", "question": "What is the verified stable version?", "expected": "4.2"},
            ],
        ),
    ]


def run_stress_demo() -> Dict[str, Any]:
    benchmark = StressBenchmark(build_default_scenarios())
    report = benchmark.run()
    chart_path = benchmark.save_chart("output/csg_validation_results.png")
    report_path = Path("output/csg_validation_results.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report["chart_path"] = str(chart_path)
    report["report_path"] = str(report_path)
    report["summary"] = {
        "overall_without_csg": report["overall_without_csg"],
        "overall_with_csg": report["overall_with_csg"],
        "overall_winner": report["overall_winner"],
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    result = run_stress_demo()
    print(json.dumps({
        "evaluation_kind": result["evaluation_kind"],
        "limitations": result["limitations"],
        "overall_without_csg": round(result["overall_without_csg"], 3),
        "overall_with_csg": round(result["overall_with_csg"], 3),
        "overall_winner": result["overall_winner"],
        "total_questions": result["total_questions"],
        "csg_improvements": result["csg"]["improvements"],
        "csg_regressions": result["csg"]["regressions"],
        "chart_path": result["chart_path"],
        "report_path": result["report_path"],
    }, indent=2))
