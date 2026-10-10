// Guardiã AI — ditado do relato.
// 1) Web Speech API (Chrome/Edge/Safari): resultados provisórios em cinza, finais entram no texto.
// 2) Fallback: MediaRecorder grava blocos de ~6 s e envia para /transcrever (faster-whisper no servidor).
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
  let listening = false, rec = null, recorder = null, stream = null, chunkTimer = null;

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

  // ── Fallback: gravação em blocos + /transcrever ─────────────────────────────
  async function startRecorder(note) {
    if (!navigator.mediaDevices || !window.MediaRecorder) { setStatus('Este navegador não suporta ditado.'); setListening(false); return; }
    const st = await fetch('/api/status').then(r => r.json()).catch(() => ({}));
    if (!st.speech_fallback) { setStatus('Ditado indisponível: o navegador não tem reconhecimento de voz e o servidor está sem faster-whisper.'); setListening(false); return; }
    try { stream = await navigator.mediaDevices.getUserMedia({ audio: true }); }
    catch (_) { setStatus('Microfone não autorizado.'); setListening(false); return; }
    const mime = MediaRecorder.isTypeSupported('audio/webm;codecs=opus') ? 'audio/webm;codecs=opus' : (MediaRecorder.isTypeSupported('audio/ogg') ? 'audio/ogg' : '');
    const startOne = () => {
      recorder = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
      const parts = [];
      recorder.ondataavailable = (e) => { if (e.data && e.data.size) parts.push(e.data); };
      recorder.onstop = async () => {
        const blob = new Blob(parts, { type: recorder.mimeType || 'audio/webm' });
        if (blob.size < 2000) return;
        setStatus('Transcrevendo o trecho…');
        const fd = new FormData(); fd.append('audio', blob, 'trecho.webm');
        try {
          const r = await fetch('/transcrever', { method: 'POST', body: fd });
          const j = await r.json();
          if (j.text) commit(j.text);
          if (listening) setStatus('Ouvindo (transcrição no servidor a cada ~6 s)…');
        } catch (_) { setStatus('Falha ao transcrever o trecho.'); }
      };
      recorder.start();
      chunkTimer = setTimeout(() => { if (recorder && recorder.state === 'recording') recorder.stop(); if (listening) startOne(); }, 6000);
    };
    setListening(true); setStatus((note ? note + ' ' : '') + 'Ouvindo (transcrição no servidor a cada ~6 s)…');
    startOne();
  }
  function stopRecorder() {
    clearTimeout(chunkTimer); chunkTimer = null;
    if (recorder && recorder.state === 'recording') recorder.stop();
    recorder = null;
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
