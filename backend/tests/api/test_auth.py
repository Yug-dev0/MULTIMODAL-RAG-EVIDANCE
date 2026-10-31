import datetime
import time
from unittest.mock import AsyncMock, patch

import jwt
import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from app.api.auth import get_google_public_keys, verify_token_fallback
import app.api.auth as auth_module

pytestmark = pytest.mark.asyncio


def _self_signed_cert_and_key():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "securetoken.system.gserviceaccount.com")])
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=1))
        .not_valid_after(datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=1))
        .sign(key, hashes.SHA256())
    )
    cert_pem = cert.public_bytes(serialization.Encoding.PEM).decode("utf-8")
    key_pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    return cert_pem, key_pem


async def test_get_google_public_keys_uses_fixed_securetoken_endpoint():
    # Regression test: the endpoint was previously built as
    # "securetoken-{project_id}", which 400s. Firebase ID tokens are always
    # signed by the fixed "securetoken@system.gserviceaccount.com" service
    # account, independent of which project issued the token.
    auth_module.PUBLIC_KEYS_CACHE = {}
    auth_module.PUBLIC_KEYS_EXPIRE = 0
    fake_resp = AsyncMock()
    fake_resp.status_code = 200
    fake_resp.json = lambda: {"some-kid": "cert-pem"}
    with patch("httpx.AsyncClient.get", new=AsyncMock(return_value=fake_resp)) as mock_get:
        await get_google_public_keys("any-project-id")
    called_url = mock_get.call_args[0][0]
    assert called_url == "https://www.googleapis.com/robot/v1/metadata/x509/securetoken@system.gserviceaccount.com"


async def test_verify_token_fallback_parses_x509_certificate_and_verifies_signature():
    # Regression test: the Google endpoint returns PEM-encoded X.509
    # certificates, not raw public keys -- passing the certificate PEM
    # directly to PyJWT used to fail with "Could not parse the provided
    # public key."
    cert_pem, private_key_pem = _self_signed_cert_and_key()
    project_id = "rag-project-230af"
    token = jwt.encode(
        {
            "sub": "uid-123",
            "email": "person@example.com",
            "aud": project_id,
            "iss": f"https://securetoken.google.com/{project_id}",
            "iat": int(time.time()),
            "exp": int(time.time()) + 3600,
        },
        private_key_pem,
        algorithm="RS256",
        headers={"kid": "test-kid"},
    )

    auth_module.PUBLIC_KEYS_CACHE = {"test-kid": cert_pem}
    auth_module.PUBLIC_KEYS_EXPIRE = time.time() + 3600

    decoded = await verify_token_fallback(token, project_id)

    assert decoded["sub"] == "uid-123"
    assert decoded["email"] == "person@example.com"
