from csg_demo import (
    ContinuousSynapticGrafting,
    KnowledgeFact,
    KnowledgeAdapter,
    StaticCore,
)


def test_no_conflict_new_fact_is_added_to_core():
    core = StaticCore({"head_of_state": "Alicia Hart"})
    adapter = KnowledgeAdapter(
        [
            KnowledgeFact(
                key="head_of_state",
                statement="Alicia Hart remains CEO of Northwind",
                value="Alicia Hart",
                confidence=0.91,
                source="internal_ledger",
            )
        ]
    )

    merged = ContinuousSynapticGrafting(core, adapter).graft()

    assert merged["head_of_state"] == "Alicia Hart"
    assert merged["head_of_state"] == core["head_of_state"]
    assert len(ContinuousSynapticGrafting(core, adapter).conflicts) == 0


def test_conflict_is_detected_and_logged():
    core = StaticCore({"current_president": "Joe Biden"})
    adapter = KnowledgeAdapter(
        [
            KnowledgeFact(
                key="current_president",
                statement="The current president is Donald Trump.",
                value="Donald Trump",
                confidence=0.88,
                source="2026_newswire",
            )
        ]
    )

    system = ContinuousSynapticGrafting(core, adapter)
    merged = system.graft()

    assert merged["current_president"] == "Donald Trump"
    assert any(item["key"] == "current_president" for item in system.conflicts)
    assert any(item["resolution"] == "adapter-wins" for item in system.conflicts)


def test_question_answer_uses_updated_memory():
    core = StaticCore({"current_president": "Joe Biden"})
    adapter = KnowledgeAdapter(
        [
            KnowledgeFact(
                key="current_president",
                statement="The current president is Donald Trump.",
                value="Donald Trump",
                confidence=0.91,
                source="2026_newswire",
            )
        ]
    )

    system = ContinuousSynapticGrafting(core, adapter)
    answer = system.answer("Who is the current president?")

    assert answer == "Donald Trump"
