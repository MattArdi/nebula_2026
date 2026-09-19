import { useState } from "react";
import Logo from "./Logo.jsx";

function EyeIcon({ open }) {
  return open ? (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7Z"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <circle cx="12" cy="12" r="3" stroke="currentColor" strokeWidth="1.5" />
    </svg>
  ) : (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d="M3 3l18 18M10.6 10.6a3 3 0 0 0 4.24 4.24M9.36 5.11A10.4 10.4 0 0 1 12 5c6.5 0 10 7 10 7a13.2 13.2 0 0 1-3.22 3.94M6.6 6.6C4.02 8.28 2 12 2 12s3.5 7 10 7a10.4 10.4 0 0 0 4.02-.8"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export default function LoginPage({ onLogin }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState(null);
  const [submitting, setSubmitting] = useState(false);

  function handleSubmit(e) {
    e.preventDefault();
    const user = username.trim();
    const pass = password.trim();
    if (!user || !pass) {
      setError("Enter a username and password to continue.");
      return;
    }
    setError(null);
    setSubmitting(true);
    // Brief delay so sign-in reads as an action, not an instant no-op.
    setTimeout(() => onLogin(user), 450);
  }

  return (
    <div className="min-h-dvh bg-surface-page flex items-center justify-center px-4 py-10 relative overflow-hidden">
      <div className="pointer-events-none absolute inset-0 overflow-hidden">
        <div className="absolute -top-24 -left-24 w-72 h-72 rounded-full bg-series-blue/10 blur-3xl" />
        <div className="absolute -bottom-24 -right-24 w-80 h-80 rounded-full bg-series-violet/10 blur-3xl" />
      </div>

      <div className="relative w-full max-w-[400px]">
        <div className="flex flex-col items-center mb-8">
          <Logo height={120} className="mb-4" />
          <h1 className="text-lg font-semibold text-ink-primary text-center">Pawl Patrol</h1>
        </div>

        <form
          onSubmit={handleSubmit}
          className="rounded-xl border border-line-border bg-surface-card px-6 py-7 space-y-4 shadow-xl shadow-black/30"
        >
          <div>
            <label htmlFor="username" className="block text-xs font-medium text-ink-secondary mb-1.5">
              Username
            </label>
            <input
              id="username"
              type="text"
              autoFocus
              autoComplete="username"
              value={username}
              onChange={(e) => {
                setUsername(e.target.value);
                setError(null);
              }}
              placeholder="e.g. inspector01"
              className="w-full rounded-md border border-line-border bg-surface-raised px-3 py-2.5 text-[15px] text-ink-primary placeholder:text-ink-muted focus:outline-none focus:ring-2 focus:ring-series-blue/50 focus:border-series-blue/60 transition-colors"
            />
          </div>

          <div>
            <label htmlFor="password" className="block text-xs font-medium text-ink-secondary mb-1.5">
              Password
            </label>
            <div className="relative">
              <input
                id="password"
                type={showPassword ? "text" : "password"}
                autoComplete="current-password"
                value={password}
                onChange={(e) => {
                  setPassword(e.target.value);
                  setError(null);
                }}
                placeholder="••••••••"
                className="w-full rounded-md border border-line-border bg-surface-raised px-3 py-2.5 pr-10 text-[15px] text-ink-primary placeholder:text-ink-muted focus:outline-none focus:ring-2 focus:ring-series-blue/50 focus:border-series-blue/60 transition-colors"
              />
              <button
                type="button"
                onClick={() => setShowPassword((v) => !v)}
                className="absolute right-2.5 top-1/2 -translate-y-1/2 text-ink-muted hover:text-ink-secondary transition-colors"
                aria-label={showPassword ? "Hide password" : "Show password"}
              >
                <EyeIcon open={showPassword} />
              </button>
            </div>
          </div>

          {error && (
            <div className="text-xs text-status-critical bg-status-critical/10 border border-status-critical/30 rounded-md px-3 py-2">
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={submitting}
            className="w-full inline-flex items-center justify-center gap-2 rounded-md bg-series-blue px-4 py-2.5 text-sm font-medium text-white transition-colors hover:bg-series-blue/85 disabled:opacity-60 disabled:cursor-not-allowed"
          >
            {submitting ? (
              <>
                <svg className="animate-spin" width="14" height="14" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                  <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="3" opacity="0.25" />
                  <path d="M21 12a9 9 0 0 0-9-9" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
                </svg>
                Signing in…
              </>
            ) : (
              "Sign in"
            )}
          </button>
        </form>

        <p className="text-center text-[11px] text-ink-muted mt-5">Any username or password works.</p>
      </div>
    </div>
  );
}
