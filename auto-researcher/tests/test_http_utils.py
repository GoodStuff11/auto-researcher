from unittest.mock import MagicMock, patch

from auto_researcher.http_utils import DEFAULT_USER_AGENT, request_with_retry


def test_returns_successful_response():
    fake_resp = MagicMock(status_code=200)
    with patch("auto_researcher.http_utils.requests.request", return_value=fake_resp) as mock_req:
        resp = request_with_retry("GET", "https://example.com")
    assert resp.status_code == 200
    mock_req.assert_called_once()


def test_retries_on_429_then_succeeds():
    fail_resp = MagicMock(status_code=429)
    ok_resp = MagicMock(status_code=200)
    with patch(
        "auto_researcher.http_utils.requests.request",
        side_effect=[fail_resp, ok_resp],
    ):
        with patch("auto_researcher.http_utils.time.sleep"):
            resp = request_with_retry(
                "GET", "https://example.com", max_retries=3, backoff_seconds=0.01
            )
    assert resp.status_code == 200


def test_sends_browser_user_agent_by_default():
    fake_resp = MagicMock(status_code=200)
    with patch("auto_researcher.http_utils.requests.request", return_value=fake_resp) as mock_req:
        request_with_retry("GET", "https://example.com")
    assert mock_req.call_args.kwargs["headers"]["User-Agent"] == DEFAULT_USER_AGENT


def test_caller_headers_are_merged_not_replaced():
    fake_resp = MagicMock(status_code=200)
    with patch("auto_researcher.http_utils.requests.request", return_value=fake_resp) as mock_req:
        request_with_retry("GET", "https://example.com", headers={"x-api-key": "secret"})
    sent_headers = mock_req.call_args.kwargs["headers"]
    assert sent_headers["x-api-key"] == "secret"
    assert sent_headers["User-Agent"] == DEFAULT_USER_AGENT
