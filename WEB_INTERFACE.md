# JARVIS web interface

Run from D:\JARVIS using the same Python environment as the desktop assistant:

```powershell
python -m pip install -r requirements-web.txt
python run_web.py
```

The launcher installs frontend packages from the lockfile when needed, starts FastAPI on 127.0.0.1:8765 and Next.js on 127.0.0.1:3000, and opens the browser. You can also double-click `run_jarvis.bat` or `run_jarvis_web.bat`. Click **Start JARVIS** to initialize the existing brain and speech workers. Ctrl+C in the launcher stops both servers. Do not run the Tkinter assistant at the same time: both would capture the same microphone.

The frontend preserves the Tkinter black/cyan rings, telemetry and conversation view. Listening, thinking and speaking status and the latest latency report appear at the top right. **System report** opens recorded response averages and a JSON download. Reports include the visible conversation; existing timing history includes previous desktop sessions. Unknown metrics display a dash rather than fabricated values.

The PC running FastAPI captures microphone audio and plays the existing Edge TTS output. This is a local desktop interface in the browser; it does not capture audio from a separate phone or remote browser. Text messages enter the same `transcript_q` as STT and go through the existing dispatcher, orchestration, memory and tools. Stop speaking or Space outside the input calls the existing interrupt mechanism. The backend starts in standby, so opening the dashboard alone does not activate the microphone or make model requests.

For separate terminals:

```powershell
python -m uvicorn assistant.web.app:app --host 127.0.0.1 --port 8765
cd frontend
npm ci
npm run dev
```

For a production frontend, run `npm run build` then `npm start` in `frontend`, keeping FastAPI running on port 8765. Use one backend process; multiple workers would create multiple assistant buses. Next proxies HTTP `/api/*` requests; live dashboard updates use a reconnecting WebSocket to port 8765. Both servers bind to loopback. No cloud hosting or credentials in the frontend are needed.

The original `python main.py` Tkinter entry point remains available.

## Voice display and live transcript

The center voice stage uses a blue/green/gold particle sphere inspired by the supplied reference. It rotates slowly from left to right while idle and reacts to measured microphone or playback levels with particle movement and restrained waves. Latest Response and speaking controls sit in the left rail. Assistant Status sits at the top right, with the plain sequential transcript below it beside the sphere. The transcript follows the latest line automatically.

Assistant words are revealed by Edge TTS WordBoundary metadata as audio playback reaches each boundary, rather than by a fixed-speed character animation. Each prefetched MP3 carries its own timing metadata. Interruptions freeze unreached words. Device buffering and browser delivery can introduce a small delay; this is not a guarantee of sample-perfect audiovisual synchronization.

For live input words, click **Enable live words**, allow browser microphone access, and choose English, Telugu, Hindi, or Tamil. Browser speech recognition supplies interim display captions only. The existing Gemini STT result remains authoritative for the brain and replaces the interim caption when it arrives. Caption availability and delay depend on browser speech recognition support and its network service. Captions pause during assistant output to avoid showing its own voice as user speech. When live captions are unavailable, the final Gemini transcript still appears after the speech segment ends.
