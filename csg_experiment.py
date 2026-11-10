from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class KnowledgeFact:
    key: str
    value: str
    confidence: float = 0.5
    source: str = "synthetic"


class SyntheticModel:
    def __init__(self, knowledge: Optional[Dict[str, str]] = None):
        self.knowledge = dict(knowledge or {})

    def answer(self, question: str) -> str:
        question_text = re.sub(r"[^a-z0-9]+", " ", question.lower()).strip()
        for key, value in self.knowledge.items():
            key_text = re.sub(r"[^a-z0-9]+", " ", key.lower()).strip()
            if not key_text:
                continue
            if key_text in question_text or any(token in question_text for token in key_text.split()):
                return str(value)
        return "unknown"


class CSGExperiment:
    def __init__(
        self,
        questions: Iterable[Dict[str, str]],
        baseline_model: SyntheticModel,
        csg_model: SyntheticModel,
    ):
        self.questions = list(questions)
        self.baseline_model = baseline_model
        self.csg_model = csg_model

    def _score_model(self, model: SyntheticModel) -> Dict[str, Any]:
        correct = 0
        per_question: List[Dict[str, Any]] = []

        for item in self.questions:
            question = str(item["question"])
            expected = str(item["expected"])
            actual = str(model.answer(question))
            is_correct = actual.lower() == expected.lower()
            if is_correct:
                correct += 1

            per_question.append(
                {
                    "question": question,
                    "expected": expected,
                    "actual": actual,
                    "correct": is_correct,
                }
            )

        score = correct / len(self.questions) if self.questions else 0.0
        return {
            "accuracy": score,
            "correct": correct,
            "total": len(self.questions),
            "per_question": per_question,
        }

    def run(self) -> Dict[str, Any]:
        without = self._score_model(self.baseline_model)
        with_csg = self._score_model(self.csg_model)

        if with_csg["accuracy"] > without["accuracy"]:
            winner = "with_csg"
        elif without["accuracy"] > with_csg["accuracy"]:
            winner = "without_csg"
        else:
            winner = "tie"

        return {
            "without_csg": without["accuracy"],
            "with_csg": with_csg["accuracy"],
            "winner": winner,
            "without_csg_details": without,
            "with_csg_details": with_csg,
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

        labels = ["Without CSG", "With CSG"]
        values = [report["without_csg"], report["with_csg"]]

        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.bar(labels, values, color=["#b8c2cc", "#4c78a8"])
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Accuracy")
        ax.set_title("CSG vs. Baseline Knowledge Accuracy")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.03,
                f"{value:.0%}",
                ha="center",
                va="bottom",
            )

        fig.tight_layout()
        fig.savefig(output_file, dpi=150)
        plt.close(fig)
        return output_file


def run_demo() -> Dict[str, Any]:
    baseline = SyntheticModel(
        {
            "current_president": "Joe Biden",
            "official_language": "English",
            "project_phase": "v1",
        }
    )
    csg_model = SyntheticModel(
        {
            "current_president": "Donald Trump",
            "official_language": "English",
            "project_phase": "beta",
        }
    )

    questions = [
        {"question": "Who is the current president?", "expected": "Donald Trump"},
        {"question": "What is the official language?", "expected": "English"},
        {"question": "What is the project phase?", "expected": "beta"},
    ]

    experiment = CSGExperiment(
        questions=questions,
        baseline_model=baseline,
        csg_model=csg_model,
    )
    report = experiment.run()
    chart_path = experiment.save_chart("output/csg_comparison.png")
    report["chart_path"] = str(chart_path)
    return report


if __name__ == "__main__":
    report = run_demo()
    print("Baseline accuracy:", f"{report['without_csg']:.0%}")
    print("CSG accuracy:", f"{report['with_csg']:.0%}")
    print("Winner:", report["winner"])
    print("Chart saved to:", report["chart_path"])
