import os
import logging
import time
from fastapi import Header, HTTPException, status
from pydantic import BaseModel
import jwt
import httpx
from cryptography import x509
import firebase_admin
from firebase_admin import credentials, auth

logger = logging.getLogger(__name__)

# Check if mock auth mode is enabled
has_config = (
    os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON")
    or os.environ.get("FIREBASE_PROJECT_ID")
    or os.environ.get("GOOGLE_CLOUD_PROJECT")
)
default_mock = "true" if not has_config else "false"
MOCK_AUTH = os.environ.get("FIREBASE_AUTH_MOCK", default_mock).lower() in ("true", "1", "yes")

firebase_initialized = False
PROJECT_ID = os.environ.get("FIREBASE_PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT")

class User(BaseModel):
    uid: str
    email: str | None = None

# Initialize Firebase Admin
if not MOCK_AUTH:
    try:
        # Check if already initialized (could be done in tests or other contexts)
        if not firebase_admin._apps:
            service_account_path = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON")
            if service_account_path and os.path.exists(service_account_path):
                cred = credentials.Certificate(service_account_path)
                firebase_admin.initialize_app(cred)
                firebase_initialized = True
                logger.info("Firebase Admin initialized via service account JSON.")
            elif PROJECT_ID:
                # If we have a project ID but no service account, we can initialize firebase admin
                # but we will use standard JWT verification as a fallback if verify_id_token complains about credentials.
                try:
                    firebase_admin.initialize_app(options={"projectId": PROJECT_ID})
                    firebase_initialized = True
                    logger.info(f"Firebase Admin initialized with project ID options: {PROJECT_ID}")
                except Exception as ex:
                    logger.warning(f"Default Firebase Admin init failed (using fallback verification): {ex}")
            else:
                firebase_admin.initialize_app()
                firebase_initialized = True
                logger.info("Firebase Admin initialized via default credentials.")
    except Exception as e:
        logger.error(f"Failed to initialize Firebase Admin: {e}. Falling back to standard JWT verification.")

# Public keys cache for verifying Firebase JWT signatures without Admin credentials
PUBLIC_KEYS_CACHE = {}
PUBLIC_KEYS_EXPIRE = 0

async def get_google_public_keys(project_id: str) -> dict:
    global PUBLIC_KEYS_EXPIRE, PUBLIC_KEYS_CACHE
    now = time.time()

    # Cache keys for 1 hour to prevent constant HTTP requests
    if not PUBLIC_KEYS_CACHE or now > PUBLIC_KEYS_EXPIRE:
        # Firebase ID tokens are always signed by this fixed service account,
        # regardless of which project is verifying them — it is NOT
        # parameterized by project_id. The previous URL
        # (.../securetoken-{project_id}) isn't a real Google endpoint, so
        # every real (non-mock) token verification failed at the certificate
        # fetch (400 from Google) before ever reaching signature checking.
        url = "https://www.googleapis.com/robot/v1/metadata/x509/securetoken@system.gserviceaccount.com"
        async with httpx.AsyncClient() as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                PUBLIC_KEYS_CACHE = resp.json()
                PUBLIC_KEYS_EXPIRE = now + 3600  # Cache for 1 hour
                logger.info("Successfully fetched and cached Google public certificates.")
            else:
                raise Exception(f"Failed to fetch public certificates from Google: status {resp.status_code}")
                
    return PUBLIC_KEYS_CACHE

async def verify_token_fallback(token: str, project_id: str) -> dict:
    """
    Verifies a Firebase ID token using public Google certificates.
    Avoids requiring Firebase Admin credentials/service accounts for ID token verification.
    """
    try:
        header = jwt.get_unverified_header(token)
        kid = header.get("kid")
        if not kid:
            raise Exception("No 'kid' header found in JWT")
            
        public_keys = await get_google_public_keys(project_id)
        if kid not in public_keys:
            raise Exception(f"Key ID '{kid}' not found in cached Google public keys")

        # Google's endpoint returns full PEM-encoded X.509 certificates, not
        # bare public keys — PyJWT needs the RSA public key extracted from
        # the certificate before it can verify an RS256 signature with it.
        cert = x509.load_pem_x509_certificate(public_keys[kid].encode("utf-8"))
        public_key = cert.public_key()

        # Decode and verify RS256 token signature, issuer, audience and expiration
        decoded = jwt.decode(
            token,
            public_key,
            algorithms=["RS256"],
            audience=project_id,
            issuer=f"https://securetoken.google.com/{project_id}"
        )
        return decoded
    except Exception as e:
        logger.error(f"Fallback JWT verification failed: {e}")
        raise e

async def get_current_user(authorization: str = Header(None)) -> User:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Authorization header. Expected Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = authorization.split("Bearer ")[1]
    
    if MOCK_AUTH:
        # In Mock Mode, decode token as an unverified JWT (or fallback to dummy parser)
        try:
            payload = jwt.decode(token, options={"verify_signature": False})
            uid = payload.get("sub") or payload.get("user_id") or payload.get("uid") or "mock-user-123"
            email = payload.get("email") or "developer@example.com"
            return User(uid=uid, email=email)
        except Exception:
            if "@" in token:
                return User(uid=f"mock-uid-{token.split('@')[0]}", email=token)
            return User(uid="mock-uid-default", email="developer@example.com")
            
    # Real Authentication
    try:
        # 1. Try Firebase Admin first if initialized with credentials
        if firebase_initialized and os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON"):
            try:
                decoded_token = auth.verify_id_token(token)
                uid = decoded_token.get("uid")
                email = decoded_token.get("email")
                return User(uid=uid, email=email)
            except Exception as admin_err:
                logger.warning(f"Firebase Admin verify failed: {admin_err}. Trying fallback verification.")
        
        # 2. Fallback to direct public key verification (does not require service account file)
        if PROJECT_ID:
            decoded_token = await verify_token_fallback(token, PROJECT_ID)
            uid = decoded_token.get("sub") or decoded_token.get("user_id")
            email = decoded_token.get("email")
            return User(uid=uid, email=email)
        else:
            raise Exception("No Firebase project ID configured to verify token.")
            
    except Exception as e:
        logger.error(f"Authentication token verification failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )
