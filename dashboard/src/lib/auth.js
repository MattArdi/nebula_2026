// Client-side-only gate — there's no backend to authenticate against, so
// this just remembers who "signed in" so the dashboard stays reachable
// across reloads until they explicitly log out.
const AUTH_KEY = "ps3_auth_user";

export function getAuthedUser() {
  try {
    return localStorage.getItem(AUTH_KEY);
  } catch {
    return null;
  }
}

export function login(username) {
  try {
    localStorage.setItem(AUTH_KEY, username);
  } catch {
    // localStorage unavailable (e.g. private browsing) — sign-in still
    // works for this tab via the caller's own state, it just won't persist.
  }
}

export function logout() {
  try {
    localStorage.removeItem(AUTH_KEY);
  } catch {
    // no-op
  }
}
