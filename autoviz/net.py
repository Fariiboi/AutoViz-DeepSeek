"""Optional TLS verification bypass for local proxies that intercept HTTPS.

Enabled only when ``DISABLE_SSL_VERIFY=true``. It patches httpx (used by the
OpenAI/DeepSeek client) and grpclib (used by the Modal SDK) so local development
works behind a TLS-intercepting proxy. Never enable this in production.
"""

from __future__ import annotations

import logging
import os
import ssl

logger = logging.getLogger(__name__)

_PATCHED = False


def configure_insecure_ssl(enabled: bool) -> None:
    """Disable certificate verification for httpx and grpclib clients."""
    global _PATCHED
    if not enabled:
        return

    # Requested environment-level escape hatches (curl / urllib based clients).
    os.environ["CURL_CA_BUNDLE"] = ""
    os.environ["PYTHONHTTPSVERIFY"] = "0"
    ssl._create_default_https_context = ssl._create_unverified_context  # type: ignore[assignment]

    if _PATCHED:
        return

    try:
        import httpx
    except ImportError:  # pragma: no cover - httpx is a project dependency
        httpx = None  # type: ignore[assignment]

    if httpx is not None:
        original_client_init = httpx.Client.__init__
        original_async_init = httpx.AsyncClient.__init__

        def _client_init(self, *args, **kwargs):
            kwargs["verify"] = False
            return original_client_init(self, *args, **kwargs)

        def _async_client_init(self, *args, **kwargs):
            kwargs["verify"] = False
            return original_async_init(self, *args, **kwargs)

        httpx.Client.__init__ = _client_init  # type: ignore[method-assign]
        httpx.AsyncClient.__init__ = _async_client_init  # type: ignore[method-assign]

    try:
        import requests
    except ImportError:  # pragma: no cover - installed via huggingface-hub
        requests = None  # type: ignore[assignment]

    if requests is not None:
        original_request = requests.Session.request

        def _request(self, *args, **kwargs):
            kwargs["verify"] = False
            return original_request(self, *args, **kwargs)

        requests.Session.request = _request  # type: ignore[method-assign]

    try:
        import grpclib.client
    except ImportError:  # pragma: no cover - installed with modal
        grpclib = None  # type: ignore[assignment]

    if grpclib is not None:
        original_channel_init = grpclib.client.Channel.__init__

        def _channel_init(self, *args, **kwargs):
            if kwargs.get("ssl") is True:
                context = ssl.create_default_context()
                context.check_hostname = False
                context.verify_mode = ssl.CERT_NONE
                kwargs["ssl"] = context
            return original_channel_init(self, *args, **kwargs)

        grpclib.client.Channel.__init__ = _channel_init  # type: ignore[method-assign]

    _PATCHED = True
    logger.warning(
        "TLS certificate verification is DISABLED (DISABLE_SSL_VERIFY=true). "
        "Use only behind a trusted local proxy."
    )
