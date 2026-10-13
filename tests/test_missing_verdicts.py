"""A response without a verdict, or a fail-open result, must not read as a clean pass."""

from __future__ import annotations

from unittest.mock import MagicMock

import httpx
import pytest
import respx

from guardex.classifier import LlamaGuardClassifier
from guardex.exceptions import GuardExAPIError, GuardExViolation
from guardex.guard import Guard, _parse_grounding_result, _parse_screen_result
from guardex.policy import GuardExPolicy
from guardex.stream import StreamGuard

BASE = "http://localhost:8001"


class TestMissingVerdictFailsClosed:
    def test_screen_without_safe_blocks(self) -> None:
        r = _parse_screen_result({"pii": {"has_pii": False}}, "input", "hi")
        assert r.classify.safe is False
        assert r.action == "block"

    def test_screen_without_has_pii_reports_pii(self) -> None:
        r = _parse_screen_result({"classify": {"safe": True}}, "input", "hi", pii_action="block")
        assert r.pii.has_pii is True
        assert r.action == "block"

    def test_scope_without_allowed_blocks(self) -> None:
        raw = {"classify": {"safe": True}, "pii": {"has_pii": False}, "scope": {}}
        r = _parse_screen_result(raw, "input", "hi")
        assert r.scope is not None and r.scope.allowed is False
        assert r.action == "block"

    def test_grounding_without_verdict_is_not_grounded(self) -> None:
        r = _parse_grounding_result({"details": [{"sentence": "a"}]})
        assert r.grounded is False
        assert r.has_contradiction is True
        assert r.details[0].grounded is False

    @respx.mock(base_url=BASE)
    def test_classify_without_safe_is_unsafe(self, respx_mock: respx.MockRouter, guard: Guard) -> None:
        respx_mock.post("/v1/classify").respond(json={"category": None})
        assert guard.classify("hi").safe is False

    @respx.mock(base_url=BASE)
    def test_pii_scan_without_has_pii_reports_pii(self, respx_mock: respx.MockRouter, guard: Guard) -> None:
        respx_mock.post("/v1/pii/scan").respond(json={"entities": []})
        assert guard.pii_scan("hi").has_pii is True

    @pytest.mark.parametrize("response", [
        {"pii": {"has_pii": False}, "text": "hi"},
        {"classify": {"safe": True}, "pii": {"has_pii": False}, "scope": {}, "text": "hi"},
    ])
    def test_stream_raises(self, response: dict) -> None:
        client = MagicMock()
        client.screen = MagicMock(return_value=(response, None))
        sg = StreamGuard(client, GuardExPolicy(api_key="gx_test_x", base_url=BASE), gate="output")
        with pytest.raises(GuardExViolation):
            list(sg.run(iter(["hi"])))

    def test_classifier_without_safe_is_unsafe(self) -> None:
        with respx.mock(base_url=BASE) as mock:
            mock.post("/v1/classify").respond(json={"category": None})
            assert LlamaGuardClassifier(GuardExPolicy(base_url=BASE)).classify(["hi"]).safe is False


class TestFailOpenIsMarked:
    def test_screen_nested_results(self) -> None:
        r = _parse_screen_result({"_fail_open": True}, "input", "hi")
        assert r.degraded and r.classify.degraded and r.pii.degraded

    @respx.mock(base_url=BASE)
    def test_classify(self, respx_mock: respx.MockRouter, guard_fail_open: Guard) -> None:
        respx_mock.post("/v1/classify").mock(side_effect=httpx.ConnectError("refused"))
        r = guard_fail_open.classify("hi")
        assert r.safe is True and r.degraded is True

    @respx.mock(base_url=BASE)
    def test_pii_scan(self, respx_mock: respx.MockRouter, guard_fail_open: Guard) -> None:
        respx_mock.post("/v1/pii/scan").mock(side_effect=httpx.ConnectError("refused"))
        r = guard_fail_open.pii_scan("hi")
        assert r.has_pii is False and r.degraded is True

    @respx.mock(base_url=BASE)
    def test_real_verdicts_are_not_degraded(self, respx_mock: respx.MockRouter, guard: Guard) -> None:
        respx_mock.post("/v1/classify").respond(json={"safe": True})
        respx_mock.post("/v1/pii/scan").respond(json={"has_pii": False, "entities": []})
        assert guard.classify("hi").degraded is False
        assert guard.pii_scan("hi").degraded is False

    def test_classifier_keeps_the_mark(self) -> None:
        with respx.mock(base_url=BASE) as mock:
            mock.post("/v1/classify").mock(side_effect=httpx.ConnectError("refused"))
            r = LlamaGuardClassifier(GuardExPolicy(base_url=BASE, fail_open=True)).classify(["a", "b"])
        assert r.safe is True and r.degraded is True


class TestLocalFailOpenIsMarked:
    def test_classifier_not_loaded(self) -> None:
        from unittest.mock import patch

        from guardex._engine.runner import LocalRunner

        with patch("guardex._engine.providers.registry.get_classifier_provider", return_value=None):
            raw = LocalRunner(fail_open=True)._classify_raw("hi")
        r = _parse_screen_result({"classify": raw, "pii": {"has_pii": False}}, "input", "hi")
        assert r.classify.safe is True and r.classify.degraded is True and r.degraded is True

    def test_pii_detector_not_loaded(self) -> None:
        from unittest.mock import patch

        from guardex._engine.runner import LocalRunner

        with patch("guardex._engine.providers.registry.get_pii_provider", return_value=None), \
             patch("guardex._engine.runner._ensure_providers"):
            raw = LocalRunner(fail_open=True).pii_scan("hi")
        assert raw["_fail_open"] is True


class TestPiiWithoutMaskedText:
    def test_screen_mask_without_masked_text_blocks(self) -> None:
        raw = {"classify": {"safe": True}, "pii": {"has_pii": True, "entities": []}, "text": "my ssn"}
        assert _parse_screen_result(raw, "input", "my ssn", pii_action="mask").action == "block"

    def test_screen_mask_without_text_blocks(self) -> None:
        raw = {"classify": {"safe": True}, "pii": {"has_pii": True, "entities": [], "masked_text": "my <SSN>"}}
        assert _parse_screen_result(raw, "input", "my ssn 123", pii_action="mask").action == "block"

    @respx.mock(base_url=BASE)
    def test_pii_mask_without_masked_text_raises(self, respx_mock: respx.MockRouter, guard: Guard) -> None:
        respx_mock.post("/v1/pii/mask").respond(json={"has_pii": True, "entities": []})
        with pytest.raises(GuardExAPIError):
            guard.pii_mask("my ssn")

    @respx.mock(base_url=BASE)
    def test_pii_mask_without_pii_returns_text(self, respx_mock: respx.MockRouter, guard: Guard) -> None:
        respx_mock.post("/v1/pii/mask").respond(json={"has_pii": False, "entities": [], "masked_text": None})
        assert guard.pii_mask("hello") == "hello"
