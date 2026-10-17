'use client';

import { useCallback, useEffect, useRef, useState } from 'react';

type Result = { isFinal: boolean; 0: { transcript: string } };
type Recognition = {
  continuous: boolean; interimResults: boolean; lang: string;
  onresult: ((event: { results: ArrayLike<Result> }) => void) | null;
  onerror: ((event: { error: string }) => void) | null;
  onend: (() => void) | null; start: () => void; stop: () => void;
};
type SpeechWindow = Window & { SpeechRecognition?: new () => Recognition; webkitSpeechRecognition?: new () => Recognition };

export function useLiveCaptions() {
  const [text, setText] = useState('');
  const [enabled, setEnabled] = useState(false);
  const [notice, setNotice] = useState('');
  const [language, setLanguage] = useState('en-IN');
  const desired = useRef(false);
  const recognition = useRef<Recognition | null>(null);
  const results = useRef<Result[]>([]);
  const consumed = useRef(0);
  const restart = useRef<ReturnType<typeof setTimeout> | null>(null);
  const paused = useRef(false);

  const stop = useCallback(() => {
    desired.current = false;
    if (restart.current) clearTimeout(restart.current);
    recognition.current?.stop(); recognition.current = null;
    setEnabled(false); setText('');
  }, []);

  const start = useCallback(() => {
    if (desired.current) return;
    const speechWindow = window as SpeechWindow;
    const Constructor = speechWindow.SpeechRecognition ?? speechWindow.webkitSpeechRecognition;
    if (!Constructor) { setNotice('Live words are unavailable in this browser. Final Gemini transcripts still work.'); return; }
    const instance = new Constructor();
    instance.continuous = true; instance.interimResults = true; instance.lang = language;
    instance.onresult = event => {
      results.current = Array.from(event.results);
      setText(results.current.slice(consumed.current).map(result => result[0].transcript).join(' ').trim());
    };
    instance.onerror = event => {
      if (['not-allowed', 'service-not-allowed', 'audio-capture', 'network'].includes(event.error)) {
        desired.current = false; setEnabled(false);
        setNotice(`Live captions unavailable (${event.error}). The main Gemini speech input is still active.`);
      }
    };
    instance.onend = () => {
      if (desired.current && !paused.current) restart.current = setTimeout(() => {
        if (!desired.current || paused.current) return;
        consumed.current = 0; results.current = [];
        try { instance.start(); } catch { desired.current = false; setEnabled(false); }
      }, 500);
    };
    recognition.current = instance; desired.current = true;
    try { instance.start(); setEnabled(true); setNotice(''); }
    catch { desired.current = false; setEnabled(false); setNotice('Allow microphone access to enable live words.'); }
  }, [language]);

  const pause = useCallback((value: boolean) => {
    if (paused.current === value) return;
    paused.current = value;
    if (!desired.current || !recognition.current) return;
    if (value) { recognition.current.stop(); setText(''); }
    else {
      consumed.current = 0; results.current = [];
      try { recognition.current.start(); } catch { /* onend may still be settling */ }
    }
  }, []);

  const acknowledge = useCallback(() => {
    let count = consumed.current;
    while (count < results.current.length && results.current[count].isFinal) count++;
    consumed.current = count;
    setText(results.current.slice(count).map(result => result[0].transcript).join(' ').trim());
  }, []);

  useEffect(() => () => {
    desired.current = false;
    if (restart.current) clearTimeout(restart.current);
    recognition.current?.stop();
  }, []);
  return { text, enabled, notice, language, setLanguage, start, stop, acknowledge, pause };
}
