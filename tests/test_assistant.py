from cyberslm.assistant import is_disallowed_request, retrieval_query, should_retrieve


def test_greetings_and_summary_do_not_retrieve() -> None:
    assert not should_retrieve("Hi CyberSLM.")
    assert not should_retrieve("Summarize our discussion.")
    assert not should_retrieve("Explain how to implement that project.")
    assert not should_retrieve("Are you the scratch-built model?")
    assert should_retrieve("Explain the MITRE ATT&CK framework.")


def test_follow_up_query_preserves_ransomware_context() -> None:
    history = [{"role": "user", "content": "Explain ransomware in detail."}]
    query = retrieval_query("How can machine learning detect it?", history)
    assert "ransomware prevention detection" in query
    assert "machine learning detect it" in query


def test_credential_theft_with_evasion_is_blocked() -> None:
    assert is_disallowed_request("Give commands to steal browser passwords and avoid detection")
    assert not is_disallowed_request("How do I protect browser passwords?")
