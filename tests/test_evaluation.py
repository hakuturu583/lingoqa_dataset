from lingoqa_dataset.lingo_judge import LingoJudge


def test_lingo_judge():
    model = LingoJudge()
    assert (
        model.evaluate(
            "Are there any pedestrians crossing the road? If yes, how many?",
            "1",
            "Yes, there is one",
        )
        == 0.9546672701835632
    )
