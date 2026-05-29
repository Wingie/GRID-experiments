/* Live presentation widgets: React (no build step, via UMD + htm tagged templates).
 *
 * Subscribes to the /events WebSocket the server exposes; pushes user questions through
 * POST /question; reads the running transcript from GET /transcript. The widgets re-render
 * whenever the server broadcasts a transcript / reward / game-frame event.
 */
(() => {
  const { useEffect, useState, useRef, useCallback } = React;
  const html = htm.bind(React.createElement);

  const wsUrl = () => (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/events';
  const post = (path, body) => fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }).then((r) => r.json());

  function useEvents() {
    const [transcript, setTranscript] = useState([]);
    const [reward, setReward] = useState({ total: 0, commentary: 0, gameplay: 0 });
    const [frame, setFrame] = useState(null);
    const [connected, setConnected] = useState(false);

    useEffect(() => {
      fetch('/transcript').then((r) => r.json()).then((d) => setTranscript(d.items || []));
      const ws = new WebSocket(wsUrl());
      ws.onopen = () => setConnected(true);
      ws.onclose = () => setConnected(false);
      ws.onerror = () => setConnected(false);
      ws.onmessage = (m) => {
        const ev = JSON.parse(m.data);
        if (ev.type === 'transcript') setTranscript((t) => [...t, ev.entry]);
        else if (ev.type === 'reward') setReward(ev.reward);
        else if (ev.type === 'frame') setFrame(ev.url);
      };
      return () => ws.close();
    }, []);

    return { transcript, reward, frame, connected };
  }

  function QuestionInput() {
    const [text, setText] = useState('');
    const [sending, setSending] = useState(false);
    const onSubmit = useCallback(async (e) => {
      e.preventDefault();
      if (!text.trim()) return;
      setSending(true);
      await post('/question', { text }).catch(() => {});
      setText('');
      setSending(false);
    }, [text]);
    return html`
      <form className="qpanel" onSubmit=${onSubmit}>
        <label>Ask the agent</label>
        <div className="row">
          <input
            type="text"
            placeholder="Why did you build the bed there?"
            value=${text}
            onChange=${(e) => setText(e.target.value)}
            disabled=${sending}
          />
          <button type="submit" disabled=${sending || !text.trim()}>Send</button>
        </div>
        <p className="hint">questions stream into the agent's commentary channel; it answers without breaking from gameplay.</p>
      </form>`;
  }

  function TranscriptStream({ entries }) {
    const ref = useRef(null);
    useEffect(() => { if (ref.current) ref.current.scrollTop = ref.current.scrollHeight; }, [entries]);
    return html`
      <div className="transcript" ref=${ref}>
        <h4>Live transcript</h4>
        ${entries.length === 0 ? html`<div className="empty">no questions yet</div>` :
          entries.slice(-12).map((e, i) => html`
            <div className="entry" key=${i}>
              <div className="q">${e.question || '(spontaneous)'}</div>
              <div className="a">${(e.response && e.response.text) || ''}</div>
              ${e.response && e.response.cited_items && e.response.cited_items.length ?
                html`<div className="cites">cited: ${e.response.cited_items.join(', ')}</div>` : null}
            </div>`)}
      </div>`;
  }

  function RewardGauge({ reward }) {
    const total = (reward.total || 0).toFixed(2);
    const commentary = (reward.commentary || 0).toFixed(2);
    const gameplay = (reward.gameplay || (reward.total - reward.commentary) || 0).toFixed(2);
    return html`
      <div className="gauge">
        <h4>Reward</h4>
        <div className="row">
          <div><span className="lbl">gameplay</span><span className="val">${gameplay}</span></div>
          <div><span className="lbl">commentary</span><span className="val">${commentary}</span></div>
          <div><span className="lbl">total</span><span className="val total">${total}</span></div>
        </div>
      </div>`;
  }

  function GameFrame({ url }) {
    return html`
      <div className="frame">
        <h4>Game</h4>
        ${url ? html`<img src=${url} alt="latest game frame" />` :
          html`<div className="placeholder">no frame yet — start the agent to stream</div>`}
      </div>`;
  }

  function StatusDot({ connected }) {
    return html`<span className=${'dot ' + (connected ? 'on' : 'off')} title=${connected ? 'live' : 'disconnected'}></span>`;
  }

  function LiveDemo() {
    const { transcript, reward, frame, connected } = useEvents();
    return html`
      <div className="live">
        <div className="bar"><${StatusDot} connected=${connected} /> events</div>
        <div className="grid">
          <${GameFrame} url=${frame} />
          <${RewardGauge} reward=${reward} />
          <${QuestionInput} />
          <${TranscriptStream} entries=${transcript} />
        </div>
      </div>`;
  }

  // Mount on slide change so we only run the WebSocket when the live slide is active.
  let root = null;
  Reveal.on('slidechanged', () => {
    const el = document.getElementById('live-demo-root');
    if (!el) return;
    const onLiveSlide = !!el.offsetParent;
    if (onLiveSlide && !root) {
      root = ReactDOM.createRoot(el);
      root.render(html`<${LiveDemo} />`);
    } else if (!onLiveSlide && root) {
      root.unmount();
      root = null;
    }
  });
  Reveal.on('ready', () => Reveal.dispatchEvent({ type: 'slidechanged' }));
})();
