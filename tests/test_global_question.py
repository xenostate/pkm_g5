from scripts.rag import is_global_question


def test_global_questions_detected():
    assert is_global_question("내 문서 전체 주제는 뭐야?")
    assert is_global_question("what are the main themes overall?")
    assert is_global_question("모든 문서를 요약해줘")
    assert is_global_question("전반적으로 어떤 내용을 공부했지?")
    assert is_global_question("Summarize across all my documents")


def test_specific_questions_not_global():
    assert not is_global_question("What is agile?")
    assert not is_global_question("RAG가 뭐야?")
    assert not is_global_question("요구사항 공학의 단계는?")
    assert not is_global_question("Explain the waterfall model")


def test_very_long_text_not_global():
    assert not is_global_question("전체 " + "x" * 300)
