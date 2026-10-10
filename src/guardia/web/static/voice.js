// Guardiã AI — ditado do relato.
// 1) Web Speech API (Chrome/Edge/Safari): resultados provisórios em cinza, finais entram no texto.
// 2) Fallback: MediaRecorder grava sem parar, corta nas pausas e envia cada trecho para /transcrever
//    (modelo de áudio via OpenRouter, whisper de reserva). Ao parar, o áudio inteiro é transcrito de novo com
//    contexto completo e substitui o texto ditado, desde que a profissional não o tenha editado no meio.
// Nada é enviado automaticamente: o texto só vai para análise quando a profissional clica em Analisar.
(function () {
  const ta = document.getElementById('relato');
  const mic = document.getElementById('mic');
  const status = document.getElementById('mic-status');
  const interim = document.getElementById('interim');
  const count = document.getElementById('charcount');
  if (!ta || !mic) return;

  const setStatus = (t) => { status.textContent = t || ''; };
  const bump = () => { if (count) count.textContent = String(ta.value.length); };
  ta.addEventListener('input', bump);
  const commit = (text) => {
    if (!text) return;
    const sep = ta.value && !/\s$/.test(ta.value) ? ' ' : '';
    ta.value = (ta.value + sep + text.trim()).slice(0, 6000);
    ta.dispatchEvent(new Event('input', { bubbles: true })); // dispara a extração ao vivo (htmx)
    ta.scrollTop = ta.scrollHeight;
  };

  // marca a ficha como editada à mão para a extração ao vivo não sobrescrever
  window.guardiaFicha = {
    reset() { const f = document.getElementById('ficha'); if (f) f.querySelectorAll('input,select').forEach(el => el.addEventListener('change', () => { const h = document.getElementById('ficha_editada'); if (h) h.value = '1'; }, { once: true })); },
  };
  window.guardiaFicha.reset();

  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  let listening = false, rec = null, recorder = null, stream = null;

  function setListening(on) {
    listening = on;
    mic.setAttribute('aria-pressed', on ? 'true' : 'false');
    mic.classList.toggle('mic-on', on);
    mic.querySelector('.mic-label').textContent = on ? 'Parar' : 'Falar';
  }

  // ── Web Speech API ──────────────────────────────────────────────────────────
  function startSR() {
    rec = new SR();
    rec.lang = 'pt-BR'; rec.continuous = true; rec.interimResults = true; rec.maxAlternatives = 1;
    rec.onresult = (e) => {
      let partial = '';
      for (let i = e.resultIndex; i < e.results.length; i++) {
        const r = e.results[i];
        if (r.isFinal) commit(r[0].transcript); else partial += r[0].transcript;
      }
      interim.textContent = partial;
    };
    rec.onerror = (e) => {
      interim.textContent = '';
      if (e.error === 'not-allowed' || e.error === 'service-not-allowed') { setStatus('Microfone não autorizado neste navegador.'); stopAll(); return; }
      if (e.error === 'network' || e.error === 'audio-capture') { // serviço de voz indisponível → tenta gravar e transcrever no servidor
        stopSR(); startRecorder('Reconhecimento do navegador indisponível; usando transcrição no servidor.');
        return;
      }
      if (e.error !== 'no-speech' && e.error !== 'aborted') setStatus('Erro no reconhecimento: ' + e.error);
    };
    rec.onend = () => { if (listening && rec) { try { rec.start(); } catch (_) {} } }; // Chrome encerra sozinho após silêncio
    rec.start();
    setListening(true); setStatus('Ouvindo… fale o relato. O texto provisório aparece em cinza.');
  }
  function stopSR() { if (rec) { const r = rec; rec = null; try { r.onend = null; r.stop(); } catch (_) {} } interim.textContent = ''; }

  // ── Fallback: gravação contínua, corte nas pausas da fala, transcrição em fila ──────────
  // Um único MediaRecorder (sem parar/reiniciar). Guardamos o segmento inicial do webm e, a cada pausa
  // detectada (ou a cada MAX_SEG s), enviamos init + blocos acumulados para /transcrever. Os textos
  // entram na ordem em que foram falados, mesmo que uma transcrição demore mais que a outra.
  const SLICE_MS = 500, SILENCE_MS = 700, MIN_SPEECH_MS = 900, MAX_SEG_MS = 15000, RMS_ON = 0.018, FINAL_MIN_SEGMENTS = 2;
  let audioCtx = null, analyser = null, vadTimer = null, initChunk = null, pending = [], segStart = 0, lastVoice = 0, hadVoice = false;
  let queue = Promise.resolve(), firstFlushDone = false;
  let allChunks = [], segments = 0, dictStart = 0, dictated = '';  // para a passada final sobre o áudio inteiro

  async function startRecorder(note) {
    if (!navigator.mediaDevices || !window.MediaRecorder) { setStatus('Este navegador não suporta ditado.'); setListening(false); return; }
    const st = await fetch('/api/status').then(r => r.json()).catch(() => ({}));
    if (!st.speech_fallback) { setStatus('Ditado indisponível: o navegador não tem reconhecimento de voz e o servidor está sem faster-whisper.'); setListening(false); return; }
    try { stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } }); }
    catch (_) { setStatus('Microfone não autorizado.'); setListening(false); return; }
    const mime = MediaRecorder.isTypeSupported('audio/webm;codecs=opus') ? 'audio/webm;codecs=opus' : (MediaRecorder.isTypeSupported('audio/ogg') ? 'audio/ogg' : '');
    recorder = new MediaRecorder(stream, mime ? { mimeType: mime, audioBitsPerSecond: 32000 } : undefined);
    initChunk = null; pending = []; segStart = Date.now(); lastVoice = 0; hadVoice = false; firstFlushDone = false;
    allChunks = []; segments = 0; dictStart = ta.value.length; dictated = '';
    recorder.ondataavailable = (e) => {
      if (!e.data || !e.data.size) return;
      if (!initChunk) { initChunk = e.data; return; }   // primeiro bloco: cabeçalho do contêiner
      pending.push(e.data); allChunks.push(e.data);
    };
    recorder.start(SLICE_MS);
    // detector de pausa: nível RMS do microfone
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const src = audioCtx.createMediaStreamSource(stream);
    analyser = audioCtx.createAnalyser(); analyser.fftSize = 1024; src.connect(analyser);
    const buf = new Float32Array(analyser.fftSize);
    vadTimer = setInterval(() => {
      analyser.getFloatTimeDomainData(buf);
      let sum = 0; for (let i = 0; i < buf.length; i++) sum += buf[i] * buf[i];
      const rms = Math.sqrt(sum / buf.length), now = Date.now();
      if (rms > RMS_ON) { lastVoice = now; hadVoice = true; }
      const segMs = now - segStart;
      const pause = hadVoice && lastVoice && (now - lastVoice) > SILENCE_MS && (lastVoice - segStart) > MIN_SPEECH_MS;
      if (pause || (hadVoice && segMs > MAX_SEG_MS)) flushSegment();
      else if (!hadVoice && segMs > MAX_SEG_MS) { pending = []; segStart = now; }  // só silêncio: descarta
    }, 100);
    setListening(true); setStatus((note ? note + ' ' : '') + 'Ouvindo… cada pausa vira um trecho transcrito.');
  }
  function flushSegment() {
    if (!initChunk || !pending.length) { segStart = Date.now(); hadVoice = false; lastVoice = 0; return; }
    const blob = new Blob([initChunk, ...pending], { type: (recorder && recorder.mimeType) || 'audio/webm' });
    pending = []; segStart = Date.now(); hadVoice = false; lastVoice = 0;
    const fd = new FormData(); fd.append('audio', blob, 'trecho.webm');
    fd.append('skip_ms', firstFlushDone ? String(SLICE_MS) : '0'); // trechos seguintes repetem o áudio do cabeçalho
    firstFlushDone = true; segments++;
    queue = queue.then(async () => {
      setStatus('Ouvindo… transcrevendo o trecho anterior.');
      try {
        const r = await fetch('/transcrever', { method: 'POST', body: fd });
        const j = await r.json();
        if (j.text) { commit(j.text); dictated = ta.value.slice(dictStart); }
      } catch (_) { setStatus('Falha ao transcrever um trecho.'); return; }
      if (listening) setStatus('Ouvindo… cada pausa vira um trecho transcrito.');
    });
  }
  // Passada final: o áudio completo, com contexto, substitui os trechos (só se o texto ditado não foi editado).
  function finalPass(init, chunks, mime) {
    if (!init || chunks.length < 2 || segments < FINAL_MIN_SEGMENTS) return;
    const blob = new Blob([init, ...chunks], { type: mime || 'audio/webm' });
    const fd = new FormData(); fd.append('audio', blob, 'completo.webm'); fd.append('skip_ms', '0');
    queue = queue.then(async () => {
      setStatus('Revisando a transcrição completa…');
      try {
        const r = await fetch('/transcrever', { method: 'POST', body: fd });
        const j = await r.json();
        const current = ta.value.slice(dictStart);
        if (j.text && current.trim() === dictated.trim()) {
          const head = ta.value.slice(0, dictStart), sep = head && !/\s$/.test(head) ? ' ' : '';
          ta.value = (head + sep + j.text.trim()).slice(0, 6000);
          ta.dispatchEvent(new Event('input', { bubbles: true }));
        }
        setStatus('');
      } catch (_) { setStatus('A revisão final da transcrição falhou; o texto ditado foi mantido.'); }
    });
  }
  function stopRecorder() {
    clearInterval(vadTimer); vadTimer = null;
    if (recorder && recorder.state === 'recording') {
      const rec0 = recorder, init = initChunk, mime = recorder.mimeType;
      rec0.onstop = () => {  // o último ondataavailable chega antes do onstop
        flushSegment(); finalPass(init, allChunks, mime);  // último trecho + revisão completa
        initChunk = null; pending = []; allChunks = [];
      };
      rec0.stop();
    } else { initChunk = null; pending = []; allChunks = []; }
    recorder = null;
    if (analyser) { try { analyser.disconnect(); } catch (_) {} analyser = null; }
    if (audioCtx) { audioCtx.close().catch(() => {}); audioCtx = null; }
    if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  }
  function stopAll() { stopSR(); stopRecorder(); setListening(false); setStatus(''); }

  mic.addEventListener('click', () => {
    if (listening) { stopAll(); return; }
    if (SR) { try { startSR(); } catch (_) { startRecorder('Reconhecimento do navegador falhou; usando o servidor.'); } }
    else startRecorder('Este navegador não tem reconhecimento de voz; usando transcrição no servidor.');
  });
  document.addEventListener('visibilitychange', () => { if (document.hidden && listening) stopAll(); });
})();
