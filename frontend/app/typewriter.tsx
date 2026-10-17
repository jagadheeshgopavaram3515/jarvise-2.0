'use client';

import { useEffect, useRef, useState } from 'react';

export function Typewriter({ text, onProgress }: { text: string; onProgress: () => void }) {
  const [visible, setVisible] = useState(0);
  const position = useRef(0);
  const previous = useRef('');
  const progress = useRef(onProgress);
  progress.current = onProgress;

  useEffect(() => {
    const length = Array.from(text).length;
    if (!text.startsWith(previous.current)) position.current = 0;
    previous.current = text;
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      position.current = length;
      setVisible(length);
      progress.current();
      return;
    }
    let frame: number;
    const startPosition = position.current;
    const started = performance.now();
    const tick = (now: number) => {
      // Advance by elapsed time so slow frames and background tabs catch up.
      const next = Math.min(length, startPosition + Math.floor((now - started) / 14));
      position.current = next;
      setVisible(next);
      progress.current();
      if (next < length) frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [text]);

  const characters = Array.from(text);
  // Never split a surrogate pair when revealing multilingual text.
  const shown = characters.slice(0, visible).join('');
  return <><span aria-hidden="true">{shown}{visible < characters.length && <span className="typing-caret">▍</span>}</span><span className="sr-only">{characters.join('')}</span></>;
}
