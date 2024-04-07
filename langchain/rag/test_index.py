from langchain_core.runnables.history import RunnableWithMessageHistory
from index import main


def test_index():
    vector_store = main(load=True)
    assert vector_store is not None


def test_main():
    rag_chain_with_source, history_store = main(load=False)

    question1 = "How to use ChatGPT in open source projects?"
    answer1 = rag_chain_with_source.invoke(
        {"question": question1}, config={"configurable": {"session_id": "123"}}
    )
    print(answer1)
    assert isinstance(answer1, dict)
    assert "context" in answer1

    question2 = "What did we just talk about?"
    answer2 = rag_chain_with_source.invoke(
        {"question": question2}, config={"configurable": {"session_id": "123"}}
    )
    print(answer2)
    assert isinstance(answer2, dict)
    assert "context" in answer2

    print(history_store)
