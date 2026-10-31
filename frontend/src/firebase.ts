import { initializeApp, getApps } from "firebase/app";
import { getAuth } from "firebase/auth";

const firebaseConfig = {
  apiKey: import.meta.env.VITE_FIREBASE_API_KEY,
  authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN,
  projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID,
  appId: import.meta.env.VITE_FIREBASE_APP_ID,
};

// Check if configuration is missing, indicating that the client should operate in Mock Auth Mode
export const IS_MOCK_AUTH = !firebaseConfig.apiKey;

let firebaseApp = null;
let firebaseAuth = null;

if (!IS_MOCK_AUTH) {
  try {
    if (!getApps().length) {
      firebaseApp = initializeApp(firebaseConfig);
    }
    firebaseAuth = getAuth();
  } catch (error) {
    console.error("Failed to initialize Firebase SDK:", error);
  }
} else {
  console.log("Firebase config not detected. Running frontend in Mock Auth Mode.");
}

export { firebaseApp, firebaseAuth };
export const auth = firebaseAuth;
