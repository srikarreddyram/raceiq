/**
 * The whole data-fetching layer: run a promise, track loading/error/data,
 * ignore results from a request that's been superseded.
 *
 * That last part is the only non-obvious piece. Every Pit Wall view
 * re-fetches when you change driver or lap, and a strategy simulation
 * takes ~1.5-2.5s server-side — long enough that clicking through three
 * drivers quickly will land the responses out of order and leave the
 * screen showing a driver you already moved away from. The generation
 * counter makes a stale response a no-op instead.
 */

import { useCallback, useEffect, useRef, useState } from "react";

export type AsyncState<T> = {
  data: T | null;
  error: string | null;
  loading: boolean;
  reload: () => void;
};

export function useAsync<T>(fn: () => Promise<T>, deps: unknown[], enabled = true): AsyncState<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(enabled);
  const [nonce, setNonce] = useState(0);
  const generation = useRef(0);

  // `fn` is a fresh closure every render; deps are what actually decide
  // when to refetch, so the lint rule's preferred "include fn" would make
  // this loop forever.
  const runner = useCallback(fn, deps);

  useEffect(() => {
    if (!enabled) {
      setLoading(false);
      return;
    }
    const current = ++generation.current;
    setLoading(true);
    setError(null);

    runner()
      .then((result) => {
        if (generation.current !== current) return; // superseded
        setData(result);
        setLoading(false);
      })
      .catch((err: unknown) => {
        if (generation.current !== current) return;
        setError(err instanceof Error ? err.message : String(err));
        setLoading(false);
      });
  }, [runner, enabled, nonce]);

  return { data, error, loading, reload: () => setNonce((n) => n + 1) };
}
