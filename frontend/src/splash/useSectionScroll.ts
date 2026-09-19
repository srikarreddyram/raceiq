/**
 * Maps scroll position within a tall container to a section index —
 * design system template §7's mechanism, PRD 13.1's "ref mirror to avoid
 * re-rendering on every scroll pixel — only 6 React renders per full
 * scroll".
 *
 * The ref mirror is the whole point: a naive `setSection(idx)` on every
 * scroll event re-renders the tree dozens of times per second for a value
 * that only changes six times. Comparing against a ref first means React
 * only hears about it when the section actually changes.
 */

import { useEffect, useRef, useState } from "react";

export function useSectionScroll(sections: number) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const [section, setSection] = useState(0);
  const sectionRef = useRef(0);
  const [progress, setProgress] = useState(0);
  const progressRef = useRef(0);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;

    const onScroll = () => {
      const rect = el.getBoundingClientRect();
      const scrolled = Math.min(1, Math.max(0, -rect.top / (rect.height - window.innerHeight)));
      const idx = Math.min(sections - 1, Math.floor(scrolled * sections));

      if (idx !== sectionRef.current) {
        sectionRef.current = idx;
        setSection(idx);
      }
      // Progress drives a thin scrub bar, so it does need finer updates —
      // but still only when it moves a visible amount (~1/300th).
      if (Math.abs(scrolled - progressRef.current) > 0.003) {
        progressRef.current = scrolled;
        setProgress(scrolled);
      }
    };

    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, [sections]);

  const jumpTo = (index: number) => {
    const el = wrapRef.current;
    if (!el) return;
    const total = el.offsetHeight - window.innerHeight;
    window.scrollTo({ top: el.offsetTop + (total * index) / sections + 10, behavior: "smooth" });
  };

  return { wrapRef, section, progress, jumpTo };
}
