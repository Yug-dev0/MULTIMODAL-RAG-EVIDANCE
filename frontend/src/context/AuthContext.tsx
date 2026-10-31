import React, { createContext, useContext, useEffect, useState, ReactNode } from "react";
import { 
  signInWithEmailAndPassword, 
  createUserWithEmailAndPassword, 
  signOut as fbSignOut, 
  signInWithPopup, 
  GoogleAuthProvider, 
  User as FirebaseUser,
  onAuthStateChanged
} from "firebase/auth";
import { auth, IS_MOCK_AUTH } from "../firebase";

export interface AppUser {
  uid: string;
  email: string | null;
  displayName?: string | null;
  photoURL?: string | null;
  isMock?: boolean;
}

interface AuthContextType {
  user: AppUser | null;
  loading: boolean;
  token: string | null;
  login: (email: string, pass: string) => Promise<void>;
  signUp: (email: string, pass: string) => Promise<void>;
  loginWithGoogle: () => Promise<void>;
  loginMock: (email: string) => void;
  logout: () => Promise<void>;
  getToken: () => Promise<string | null>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) throw new Error("useAuth must be used within an AuthProvider");
  return context;
};

interface AuthProviderProps {
  children: ReactNode;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [user, setUser] = useState<AppUser | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  // Initialize Auth state
  useEffect(() => {
    if (IS_MOCK_AUTH) {
      // In Mock Mode, read user from localStorage to persist sessions
      const storedUser = localStorage.getItem("mock_user");
      if (storedUser) {
        try {
          const parsed = JSON.parse(storedUser);
          setUser(parsed);
          setToken(`mock-token-${parsed.email}`);
        } catch {
          localStorage.removeItem("mock_user");
        }
      }
      setLoading(false);
      return;
    }

    if (!auth) return;

    // Real Firebase listener
    const unsubscribe = onAuthStateChanged(auth, async (fbUser) => {
      if (fbUser) {
        const uToken = await fbUser.getIdToken();
        setToken(uToken);
        setUser({
          uid: fbUser.uid,
          email: fbUser.email,
          displayName: fbUser.displayName,
          photoURL: fbUser.photoURL,
          isMock: false
        });
      } else {
        setUser(null);
        setToken(null);
      }
      setLoading(false);
    });

    return () => unsubscribe();
  }, []);

  const login = async (email: string, pass: string) => {
    if (IS_MOCK_AUTH) {
      loginMock(email);
      return;
    }
    if (!auth) return;
    await signInWithEmailAndPassword(auth, email, pass);
  };

  const signUp = async (email: string, pass: string) => {
    if (IS_MOCK_AUTH) {
      loginMock(email);
      return;
    }
    if (!auth) return;
    await createUserWithEmailAndPassword(auth, email, pass);
  };

  const loginWithGoogle = async () => {
    if (IS_MOCK_AUTH) {
      loginMock("google.dev@example.com");
      return;
    }
    if (!auth) return;
    const provider = new GoogleAuthProvider();
    await signInWithPopup(auth, provider);
  };

  const loginMock = (email: string) => {
    const formattedEmail = email || "developer@example.com";
    const mockUser: AppUser = {
      uid: `mock-uid-${formattedEmail.split("@")[0]}`,
      email: formattedEmail,
      displayName: formattedEmail.split("@")[0].toUpperCase(),
      photoURL: null,
      isMock: true,
    };
    setUser(mockUser);
    const mockToken = `mock-token-${formattedEmail}`;
    setToken(mockToken);
    localStorage.setItem("mock_user", JSON.stringify(mockUser));
  };

  const logout = async () => {
    if (IS_MOCK_AUTH) {
      setUser(null);
      setToken(null);
      localStorage.removeItem("mock_user");
      return;
    }
    if (!auth) return;
    await fbSignOut(auth);
  };

  const getToken = async (): Promise<string | null> => {
    if (IS_MOCK_AUTH) {
      return token;
    }
    if (!auth || !auth.currentUser) return null;
    // Force refresh the token to make sure it's valid
    const refreshedToken = await auth.currentUser.getIdToken(true);
    setToken(refreshedToken);
    return refreshedToken;
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        loading,
        token,
        login,
        signUp,
        loginWithGoogle,
        loginMock,
        logout,
        getToken
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};
