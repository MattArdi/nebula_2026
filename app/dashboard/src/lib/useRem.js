import { useEffect, useState } from "react";

const read = () => parseFloat(getComputedStyle(document.documentElement).fontSize) || 16;

// The page's current root font size in px. Chart text and dimensions are
// drawn in px, so they use this to scale with the same rule as the rest of the
// page's text (see html { font-size } in index.css).
export function useRem() {
  const [rem, setRem] = useState(read);
  useEffect(() => {
    const onResize = () => setRem(read());
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);
  return rem;
}

// px(n) turns a size designed at 16px into the current scale.
export function usePx() {
  const rem = useRem();
  return (n) => (n * rem) / 16;
}
