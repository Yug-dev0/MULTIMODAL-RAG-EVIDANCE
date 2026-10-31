import React, { useState } from "react";
import { useAuth } from "../context/AuthContext";
import { IS_MOCK_AUTH } from "../firebase";

export const Login: React.FC = () => {
  const { login, signUp, loginWithGoogle } = useAuth();
  const [tab, setTab] = useState<"signin" | "signup">("signin");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!email || !password) {
      setError("Please fill in all fields.");
      return;
    }
    setError(null);
    setLoading(true);
    try {
      if (tab === "signin") {
        await login(email, password);
      } else {
        await signUp(email, password);
      }
    } catch (err: any) {
      console.error(err);
      setError(err.message || "Authentication failed. Please check credentials.");
    } finally {
      setLoading(false);
    }
  };

  const handleGoogleSignIn = async () => {
    setError(null);
    setLoading(true);
    try {
      await loginWithGoogle();
    } catch (err: any) {
      console.error(err);
      setError(err.message || "Google Sign-In failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="w-full max-w-[560px] mx-auto my-auto">
      <div className="bracket-frame bg-panel/50 border border-hairline p-8 sm:p-12 backdrop-blur-xl shadow-2xl">
        <header className="mb-8 text-center">
          <div className="eyebrow mb-2.5 text-xs tracking-[0.25em]">Multimodal Evidence Console</div>
          <h2 className="font-sans text-3xl font-bold tracking-tight text-ink">
            System <span className="text-signal">Authorization</span>
          </h2>
        </header>

        {IS_MOCK_AUTH && (
          <div className="mb-6 border border-amber/30 bg-amber/5 p-3.5 font-mono text-xs text-amber">
            <span className="font-bold">DEVELOPMENT MOCK MODE ACTIVE:</span>
            <p className="mt-1 opacity-85">
              No Firebase environment variables found. Real auth bypassed.
            </p>
          </div>
        )}

        {error && (
          <div className="mb-6 border border-red-500/30 bg-red-500/10 p-3.5 text-xs text-red-400 font-mono">
            {error}
          </div>
        )}

        {/* Tab switcher */}
        <nav className="mb-8 flex gap-2 border-b border-hairline pb-3">
          <button
            onClick={() => setTab("signin")}
            className={`flex-1 pb-2.5 font-mono text-xs uppercase tracking-wider transition-all text-center ${
              tab === "signin" 
                ? "border-b-2 border-signal text-signal font-semibold" 
                : "text-dim hover:text-ink"
            }`}
          >
            Sign In
          </button>
          <button
            onClick={() => setTab("signup")}
            className={`flex-1 pb-2.5 font-mono text-xs uppercase tracking-wider transition-all text-center ${
              tab === "signup" 
                ? "border-b-2 border-signal text-signal font-semibold" 
                : "text-dim hover:text-ink"
            }`}
          >
            Create Account
          </button>
        </nav>

        <form onSubmit={handleSubmit} className="flex flex-col gap-5">
          <div className="flex flex-col gap-2">
            <label className="font-mono text-[11px] uppercase tracking-wider text-dim">Email Address</label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="e.g. developer@example.com"
              disabled={loading}
              className="rounded-md border border-hairline bg-panel px-4 py-3 font-sans text-[15px] text-ink placeholder:text-dim focus:outline-none focus:border-signal/80 transition-colors"
            />
          </div>

          <div className="flex flex-col gap-2">
            <label className="font-mono text-[11px] uppercase tracking-wider text-dim">Password</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••••••"
              disabled={loading}
              className="rounded-md border border-hairline bg-panel px-4 py-3 font-sans text-[15px] text-ink placeholder:text-dim focus:outline-none focus:border-signal/80 transition-colors"
            />
          </div>

          <button
            type="submit"
            disabled={loading}
            className="mt-2 rounded-md bg-signal/15 border border-signal/40 py-3.5 font-mono text-xs font-semibold uppercase tracking-widest text-signal hover:bg-signal/25 active:scale-[0.99] transition-all disabled:opacity-50"
          >
            {loading ? "Authenticating…" : tab === "signin" ? "Initialize Session" : "Provision Credentials"}
          </button>
        </form>

        <div className="my-6 flex items-center justify-between gap-4 text-dim">
          <div className="h-[1px] flex-1 bg-hairline" />
          <span className="font-mono text-[10px] uppercase tracking-widest text-dim/70">or connect via</span>
          <div className="h-[1px] flex-1 bg-hairline" />
        </div>

        <div className="flex flex-col gap-3">
          <button
            onClick={handleGoogleSignIn}
            disabled={loading}
            className="flex items-center justify-center gap-3 rounded-md border border-hairline bg-panel/60 py-3 font-mono text-xs text-ink hover:text-signal hover:bg-panel hover:border-signal/30 transition-all"
          >
            <svg className="h-4 w-4" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12.24 10.285V13.4h6.887C18.2 15.614 15.645 18 12.24 18c-3.86 0-7-3.14-7-7s3.14-7 7-7c1.7 0 3.3.61 4.5 1.76l2.45-2.45C17.7 1.83 15.09 1 12.24 1c-5.523 0-10 4.477-10 10s4.477 10 10 10c5.78 0 9.61-4.06 9.61-9.78 0-.66-.06-1.29-.17-1.935H12.24z"/>
            </svg>
            Google Identity
          </button>
        </div>
      </div>
    </div>
  );
};
