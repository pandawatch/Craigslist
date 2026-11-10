from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Dict, Iterable, List, Optional


@dataclass(frozen=True)
class KnowledgeFact:
    key: str
    statement: str
    value: str
    confidence: float = 0.5
    source: str = "unknown"
    vector: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "key", self.key.strip())
        object.__setattr__(self, "statement", self.statement.strip())
        object.__setattr__(self, "value", self.value.strip())
        object.__setattr__(self, "confidence", max(0.0, min(1.0, float(self.confidence))))

    def to_vector(self) -> Dict[str, float]:
        if self.vector:
            return dict(self.vector)

        tokens = re.findall(r"[A-Za-z0-9]+", self.statement.lower())
        return {token: self.confidence for token in tokens if token}


class StaticCore(dict):
    def __init__(self, beliefs: Optional[Dict[str, str]] = None):
        super().__init__(beliefs or {})

    def answer_for(self, key: str) -> Optional[str]:
        return self.get(key)


class KnowledgeAdapter:
    def __init__(self, facts: Optional[Iterable[KnowledgeFact]] = None):
        self.facts: List[KnowledgeFact] = list(facts or [])

    @classmethod
    def from_fact_stream(cls, facts: Iterable[KnowledgeFact]) -> "KnowledgeAdapter":
        return cls(list(facts))

    def add(self, fact: KnowledgeFact) -> None:
        self.facts.append(fact)

    def distilled_vectors(self) -> Dict[str, Dict[str, float]]:
        result: Dict[str, Dict[str, float]] = {}
        for fact in self.facts:
            result[fact.key] = fact.to_vector()
        return result


class ContinuousSynapticGrafting:
    def __init__(self, static_core: StaticCore, adapter: KnowledgeAdapter, merge_strength: float = 0.85):
        self.static_core = static_core
        self.adapter = adapter
        self.merge_strength = max(0.0, min(1.0, float(merge_strength)))
        self.conflicts: List[Dict[str, object]] = []

    def _normalize_key(self, key: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", key.lower()).strip()

    def _find_relevant_key(self, question: str) -> Optional[str]:
        question_key = self._normalize_key(question)
        for key in self.static_core:
            normalized = self._normalize_key(key)
            if normalized in question_key or question_key in normalized:
                return key
        for fact in self.adapter.facts:
            normalized = self._normalize_key(fact.key)
            if normalized in question_key or question_key in normalized:
                return fact.key
        return None

    def graft(self) -> Dict[str, str]:
        merged = dict(self.static_core)
        self.conflicts = []

        for fact in self.adapter.facts:
            key = fact.key
            new_value = fact.value

            if key not in merged:
                merged[key] = new_value
                continue

            old_value = merged[key]
            if old_value == new_value:
                continue

            resolution = "adapter-wins" if fact.confidence >= 0.8 else "core-wins"
            self.conflicts.append(
                {
                    "key": key,
                    "previous": old_value,
                    "incoming": new_value,
                    "confidence": fact.confidence,
                    "resolution": resolution,
                }
            )

            if resolution == "adapter-wins":
                merged[key] = new_value
            else:
                merged[key] = old_value

        self.static_core.clear()
        self.static_core.update(merged)
        return dict(merged)

    def answer(self, question: str) -> str:
        merged = self.graft()
        key = self._find_relevant_key(question)
        if key and key in merged:
            return str(merged[key])
        return "No answer found in the current knowledge state."


if __name__ == "__main__":
    core = StaticCore({"current_president": "Joe Biden", "project_status": "v1"})
    adapter = KnowledgeAdapter(
        [
            KnowledgeFact(
                key="current_president",
                statement="The current president is Donald Trump.",
                value="Donald Trump",
                confidence=0.91,
                source="2026_newswire",
            ),
            KnowledgeFact(
                key="project_status",
                statement="The project is now in beta with new adapter modules.",
                value="beta",
                confidence=0.74,
                source="internal_status",
            ),
        ]
    )

    system = ContinuousSynapticGrafting(core, adapter)
    print("Core beliefs before graft:", dict(system.static_core))
    merged = system.graft()
    print("Merged beliefs after graft:", merged)
    print("Conflict log:", system.conflicts)
    print("Answer example:", system.answer("Who is the current president?"))
