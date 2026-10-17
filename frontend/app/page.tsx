'use client';

import { FormEvent, useCallback, useEffect, useRef, useState } from 'react';
import { applyConversationEvents, type ConversationEvent as Event, type Message } from './conversation';
import { VoiceSphere } from './sphere';
import { useLiveCaptions } from './live-captions';

type Snapshot = {
  session_id?: string;
  audio?: { mic_level: number; output_level: number; user_speaking: boolean };
  phase: string; ready: boolean; microphone_ready: boolean; speaking: boolean; mode: string;
  events: Event[];
  system: { cpu: number; memory: number; frequency: number; uptime: number };
  report: { count: number; window: number; latest: Record<string, number | string> | null; averages: Record<string, number | null> };
};
const labels: Record<string, string> = { offline: 'Standby', initializing: 'Initializing', listening: 'Listening', thinking: 'Thinking', speaking: 'Speaking', recognizing: 'Recognizing', sleeping: 'Sleeping', error: 'Needs attention' };
const seconds = (value: unknown) => typeof value === 'number' ? `${value.toFixed(2)} s` : '—';

export default function Home() {
  const captions = useLiveCaptions();
  const [data, setData] = useState<Snapshot | null>(null);
  const [connected, setConnected] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [notice, setNotice] = useState('');
  const [input, setInput] = useState('');
  const [pending, setPending] = useState(false);
  const [reportOpen, setReportOpen] = useState(false);
  const [clock, setClock] = useState('');
  const lastId = useRef(0);
  const sessionId = useRef<string | undefined>(undefined);
  const conversation = useRef<HTMLDivElement>(null);
  const reportPanel = useRef<HTMLElement>(null);

  useEffect(() => {
    if (!reportOpen) return;
    const previous = document.activeElement as HTMLElement | null;
    function trap(event: KeyboardEvent) {
      if (event.key !== 'Tab') return;
      const controls = reportPanel.current?.querySelectorAll<HTMLElement>('button:not(:disabled), [href], input:not(:disabled)');
      if (!controls?.length) return;
      const first = controls[0], last = controls[controls.length - 1];
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
    }
    window.addEventListener('keydown', trap);
    return () => { window.removeEventListener('keydown', trap); previous?.focus(); };
  }, [reportOpen]);

  useEffect(() => {
    let disposed = false;
    let socket: WebSocket;
    let retry: ReturnType<typeof setTimeout>;
    function connect() {
      socket = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.hostname}:8765/ws`);
      socket.onopen = () => { if (!disposed) setConnected(true); };
      socket.onmessage = (message) => {
        if (disposed) return;
        const next: Snapshot = JSON.parse(message.data);
        const restarted = !!sessionId.current && next.session_id !== sessionId.current;
        if (restarted || (next.events.length && next.events[next.events.length - 1].id < lastId.current)) lastId.current = 0;
        sessionId.current = next.session_id;
        const fresh = next.events.filter(event => event.id > lastId.current);
        if (fresh.length) {
          lastId.current = fresh[fresh.length - 1].id;
          if (fresh.some(event => event.kind === 'subtitle' && event.role === 'user')) captions.acknowledge();
          setMessages(previous => applyConversationEvents(restarted ? [] : previous, fresh));
          const notifications = fresh.filter(event => event.kind === 'notification');
          if (notifications.length) setNotice(notifications[notifications.length - 1].text);
        }
        else if (restarted) setMessages([]);
        setData(next);
      };
      socket.onerror = () => socket.close();
      socket.onclose = () => { if (!disposed) { setConnected(false); retry = setTimeout(connect, 1500); } };
    }
    connect();
    const timer = setInterval(() => setClock(new Date().toLocaleTimeString('en-IN')), 1000);
    return () => { disposed = true; clearInterval(timer); clearTimeout(retry); socket?.close(); };
  }, []);

  useEffect(() => { conversation.current?.scrollTo({ top: conversation.current.scrollHeight, behavior: 'instant' }); }, [messages, captions.text]);

  useEffect(() => { captions.pause(!!data?.speaking); }, [data?.speaking, captions.pause]);

  const command = useCallback(async (path: string, body?: object) => {
    const response = await fetch(`/api/${path}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body ?? {}) });
    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(typeof error.detail === 'string' ? error.detail : 'Unable to complete request.');
    }
  }, []);

  const stop = useCallback(() => {
    if (data?.ready) command('interrupt').catch(error => setNotice(error.message));
  }, [command, data?.ready]);
  useEffect(() => {
    function keydown(event: KeyboardEvent) {
      const target = event.target as HTMLElement;
      if (event.code === 'Space' && !['INPUT', 'TEXTAREA', 'BUTTON'].includes(target.tagName) && !target.isContentEditable) {
        event.preventDefault(); stop();
      }
      if (event.key === 'Escape') setReportOpen(false);
    }
    window.addEventListener('keydown', keydown);
    return () => window.removeEventListener('keydown', keydown);
  }, [stop]);

  async function send(event: FormEvent) {
    event.preventDefault();
    if (!input.trim() || pending) return;
    setPending(true);
    try { await command('chat', { text: input.trim() }); setInput(''); }
    catch (error) { setNotice((error as Error).message); }
    finally { setPending(false); }
  }
  async function start() {
    captions.start();
    setPending(true);
    try { await command('start'); }
    catch (error) { setNotice((error as Error).message); }
    finally { setPending(false); }
  }
  function download() {
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a'); link.href = url; link.download = `jarvis-report-${new Date().toISOString().slice(0, 10)}.json`; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  const phase = connected ? data?.phase ?? 'offline' : 'offline';
  const active = connected && data?.ready;
  const metrics = data?.report.latest;
  const sphereLevel = Math.max(data?.audio?.mic_level ?? 0, data?.audio?.output_level ?? 0);

  return (
    <main className={`dashboard phase-${phase}`}>
      <header className="masthead">
        <a href="/" className="wordmark" aria-label="JARVIS home"><span className="brand-icon">J</span> J.A.R.V.I.S <small>PERSONAL ASSISTANT</small></a>
        <div className={`connection ${connected ? 'online' : ''}`}><span className="dot" />{connected ? 'LOCAL SYSTEM ONLINE' : 'CONNECTING TO BACKEND'}</div>
        <button className="report-trigger" onClick={() => setReportOpen(true)} aria-haspopup="dialog"><span>↗</span> System report</button>
      </header>

      <section className="workspace" aria-label="JARVIS control room">
        <aside className="telemetry">
          <p className="eyebrow">SYSTEM TELEMETRY <span>01</span></p>
          <div className="metric"><span>Processor</span><strong>{data ? data.system.cpu.toFixed(0) : '—'}<small>%</small></strong><div className="meter"><i style={{ width: `${data?.system.cpu ?? 0}%` }} /></div></div>
          <div className="metric"><span>Memory</span><strong>{data ? data.system.memory.toFixed(0) : '—'}<small>%</small></strong><div className="meter"><i style={{ width: `${data?.system.memory ?? 0}%` }} /></div></div>
          <dl className="system-details"><div><dt>CPU frequency</dt><dd>{data?.system.frequency || '—'} MHz</dd></div><div><dt>Speech input</dt><dd className={data?.microphone_ready ? 'green' : ''}>{data?.microphone_ready ? 'Ready' : 'Standby'}</dd></div><div><dt>Voice engine</dt><dd>Edge TTS</dd></div></dl>




          <div className="latency"><div className="panel-heading"><p className="eyebrow">LATEST RESPONSE</p><span>SECONDS</span></div><dl><div><dt>Speech recognition</dt><dd>{seconds(metrics?.stt)}</dd></div><div><dt>Brain · first response</dt><dd>{seconds(metrics?.gemini)}</dd></div><div><dt>Voice synthesis</dt><dd>{seconds(metrics?.tts_synth)}</dd></div><div className="total"><dt>Time to voice</dt><dd>{seconds(metrics?.total)}</dd></div></dl><button className="text-button" onClick={() => setReportOpen(true)}>View full report <span>↗</span></button></div>


          <div className="voice-controls">
          <div className="core-status" role="status"><span className="dot" />{labels[phase] ?? phase}<div className="sound-bars" aria-hidden="true">{Array.from({ length: 9 }, (_, i) => <i key={i} style={{ animationDelay: `${i * .11}s` }} />)}</div></div>
          <p className="core-hint">{!connected ? 'Start the FastAPI backend to connect.' : phase === 'offline' ? 'Bring your assistant online when you’re ready.' : phase === 'thinking' ? 'Working on your request.' : phase === 'speaking' ? 'Press Space to interrupt.' : phase === 'sleeping' ? 'Say “wake up” to resume.' : phase === 'initializing' ? 'Preparing the brain and speech pipeline.' : 'Speak naturally. I’m here to help.'}</p>
          <div className="core-actions">{phase === 'offline' && <button className="primary" disabled={!connected || pending} onClick={start}>Start JARVIS <span>→</span></button>}<button className="stop-button" onClick={stop} disabled={!active}><span>■</span> Stop speaking <kbd>SPACE</kbd></button></div>
          </div>
          <div className="local-note"><span className="dot" /> ON THIS DEVICE<p>Your PC’s microphone and speakers connect directly to JARVIS.</p></div>
        </aside>

        <section className="core" aria-label="Assistant activity">
          <aside className="assistant-status" aria-label="Assistant status"><div className={`status-card ${active ? 'active' : ''}`}><p className="eyebrow">ASSISTANT STATUS</p><h1><span className="dot" />{labels[phase] ?? phase}</h1><p>{active ? 'Brain connected · speech pipeline active' : 'Waiting for an active session'}</p><div className="status-steps">{['listening', 'thinking', 'speaking'].map(state => <span key={state} className={phase === state ? 'selected' : ''}>{state}</span>)}</div></div></aside>
          <div className="sphere-stage">
          <div className="core-coordinate">VOICE INTERFACE <span>•</span> JARVIS CORE</div>
          <VoiceSphere thinking={phase === 'thinking'} level={sphereLevel} label={data?.audio?.user_speaking ? 'LISTENING TO YOU' : (labels[phase] ?? phase).toUpperCase()} />

          </div>
      <section className="conversation-panel" aria-label="Conversation">
        <div className="panel-heading"><p className="eyebrow">CONVERSATION <span className="count">{messages.length.toString().padStart(2, '0')}</span></p><span>LIVE <i className="dot" /></span></div>
        <div ref={conversation} className="conversation" role="log" aria-live="polite" aria-relevant="additions text">
          {messages.map((message, index) => <div className={`message ${message.role} delivery-${message.delivery ?? 'received'}`} key={`${message.id}-${index}`}>
            <span className="speaker-label">{message.role === 'user' ? 'Me' : 'Jarvis'} :</span>
            <p>{message.role === 'user' ? message.text : <>
              {message.parts?.map(part => part.visible).filter(Boolean).join(' ') || <span className="waiting-words">{message.delivery === 'failed' ? 'Voice unavailable' : ''}</span>}
              {['speaking', 'responding'].includes(message.delivery ?? '') && <span className="typing-caret" aria-hidden="true" />}
            </>}</p>
          </div>)}
          {captions.text && <div className="message user live-input"><span className="speaker-label">Me :</span><p>{captions.text}<span className="typing-caret" aria-hidden="true" /></p></div>}
          {!messages.length && !captions.text && <div className="empty-conversation"><p>Your conversation appears here, one line after another.<small>JARVIS's words follow its voice. Enable live words to see your speech as it is recognized.</small></p></div>}
        </div>
        <div className="live-caption-controls"><button type="button" onClick={captions.enabled ? captions.stop : captions.start} disabled={!active && !captions.enabled}>{captions.enabled ? 'Live words on' : 'Enable live words'}</button><select aria-label="Live caption language" value={captions.language} disabled={captions.enabled} onChange={event => captions.setLanguage(event.target.value)}><option value="en-IN">English</option><option value="te-IN">Telugu</option><option value="hi-IN">Hindi</option><option value="ta-IN">Tamil</option></select><span>{captions.enabled ? 'Live words may be corrected when you finish speaking' : 'Microphone permission needed for live words'}</span></div>
        {captions.notice && <p className="caption-notice" role="status">{captions.notice}</p>}
        <form onSubmit={send} className="composer"><span className="prompt-mark">›</span><input aria-label="Message JARVIS" placeholder={active ? 'Ask JARVIS anything…' : 'Start JARVIS to begin a conversation…'} value={input} onChange={event => setInput(event.target.value)} disabled={!active} maxLength={8000} /><button type="submit" disabled={!active || pending || !input.trim()} aria-label="Send message">↑</button></form>
      </section>
        </section>


      </section>

      {notice && <div className="notification" role="alert"><span>{notice}</span><button aria-label="Dismiss notification" onClick={() => setNotice('')}>×</button></div>}
      <footer><span><i className="dot" /> JARVIS DESKTOP / WEB INTERFACE</span><span suppressHydrationWarning>{clock || 'LOCAL TIME'} <b>•</b> {data ? `${Math.floor(data.system.uptime / 60)} MIN UPTIME` : 'STANDBY'}</span></footer>

      {reportOpen && <div className="modal-backdrop" onClick={() => setReportOpen(false)}><section ref={reportPanel} className="report-modal" role="dialog" aria-modal="true" aria-labelledby="report-title" onClick={event => event.stopPropagation()}><div className="panel-heading"><p className="eyebrow">SYSTEM DIAGNOSTICS</p><button autoFocus onClick={() => setReportOpen(false)} aria-label="Close report">×</button></div><h2 id="report-title">Session report<span className="dot" /></h2><p className="report-description">Live system health and recorded response timing.</p><div className="report-summary"><div><small>State</small><strong>{labels[phase]}</strong></div><div><small>Recorded responses</small><strong>{data?.report.count ?? 0}</strong></div><div><small>Memory used</small><strong>{data?.system.memory ?? '—'}%</strong></div></div><table><caption>Average timing · last {data?.report.window ?? 0} recorded responses</caption><tbody>{[['Speech recognition', 'stt'], ['Brain · first response', 'gemini'], ['Voice synthesis', 'tts_synth'], ['Time to voice', 'total']].map(([title, key]) => <tr key={key}><th>{title}</th><td>{seconds(data?.report.averages[key])}</td></tr>)}</tbody></table><p className="report-note">Timing history includes previous sessions. Voice input and output use this PC. Download includes the visible conversation and diagnostics.</p><button className="primary" onClick={download} disabled={!data}>Download report <span>↓</span></button></section></div>}
    </main>
  );
}
