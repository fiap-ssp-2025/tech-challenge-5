// Guardiã AI — comportamento do indicador de estado no cabeçalho.
// Popover nativo (clique abre/fecha, clique fora fecha) + abrir ao passar o mouse e fechar ao sair.
(function () {
  let hideTimer = null;
  const pill = () => document.getElementById('status-pill');
  const pop = () => document.getElementById('status-pop');
  const show = () => { const p = pop(); clearTimeout(hideTimer); if (p && !p.matches(':popover-open')) { try { p.showPopover(); } catch (_) {} } };
  const hideSoon = () => { clearTimeout(hideTimer); hideTimer = setTimeout(() => { const p = pop(); if (p && p.matches(':popover-open')) { try { p.hidePopover(); } catch (_) {} } }, 180); };
  document.addEventListener('mouseover', (e) => {
    const wrap = pill(); if (!wrap) return;
    if (wrap.contains(e.target) || (pop() && pop().contains(e.target))) show();
  });
  document.addEventListener('mouseout', (e) => {
    const wrap = pill(); if (!wrap) return;
    const to = e.relatedTarget;
    const inside = to && (wrap.contains(to) || (pop() && pop().contains(to)));
    if (!inside && (wrap.contains(e.target) || (pop() && pop().contains(e.target)))) hideSoon();
  });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') hideSoon(); });
})();

// Painel de protocolos (spec 003): <dialog> não modal aberto com show(), Esc fecha, foco volta ao botão.
// A pergunta aparece na hora como balão provisório; a resposta do servidor (o turno inteiro) o substitui.
window.guardiaChat = (function () {
  const $ = (id) => document.getElementById(id);
  const thread = () => $('chat-thread');
  const toBottom = () => { const t = thread(); if (t) t.scrollTop = t.scrollHeight; };
  const grow = (ta) => { ta.style.height = 'auto'; ta.style.height = Math.min(ta.scrollHeight, 160) + 'px'; };
  const api = {
    focus() { const ta = $('pergunta'); if (ta) ta.focus(); },
    open() {
      const d = $('chat-drawer'); if (!d) return;
      if (!d.open) d.show();
      $('chat-fab')?.setAttribute('aria-expanded', 'true');
      toBottom(); api.focus();
    },
    close() {
      const d = $('chat-drawer'); if (!d || !d.open) return;
      d.close();
      const fab = $('chat-fab'); if (fab) { fab.setAttribute('aria-expanded', 'false'); fab.focus(); }
    },
    ask(text) {
      const ta = $('pergunta'), form = $('chat-form'); if (!ta || !form) return;
      ta.value = text.trim(); form.requestSubmit();
    },
    pending(form) {
      const ta = form.querySelector('textarea'), q = ta.value.trim(), t = thread(); if (!q || !t) return;
      const turn = document.createElement('article');
      turn.className = 'chat-turn chat-pending';
      turn.innerHTML = '<p class="chat-msg chat-user"></p><p class="chat-typing" role="status"><span></span><span></span><span></span> Consultando os protocolos…</p>';
      turn.querySelector('.chat-user').textContent = q;
      t.append(turn); toBottom();
      form.dataset.pending = q; ta.value = ''; grow(ta);
    },
    settle(form, ctx) {
      document.querySelectorAll('#chat-thread .chat-pending').forEach((el) => el.remove());
      const ok = ctx && ctx.response && ctx.response.status < 300;
      const ta = form.querySelector('textarea');
      if (!ok && form.dataset.pending && !ta.value) { ta.value = form.dataset.pending; grow(ta); }  // devolve o texto para reenviar
      delete form.dataset.pending; toBottom(); api.focus();
    },
  };
  document.addEventListener('keydown', (e) => {
    const d = $('chat-drawer');
    if (e.key === 'Escape' && d && d.open && d.contains(document.activeElement)) { e.preventDefault(); api.close(); return; }
    if (e.target && e.target.id === 'pergunta' && e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
      e.preventDefault(); $('chat-form')?.requestSubmit();
    }
  });
  document.addEventListener('input', (e) => { if (e.target && e.target.id === 'pergunta') grow(e.target); });
  // painel aberto pelo servidor (acesso direto a /protocolos): rola até o fim e foca o campo
  document.addEventListener('DOMContentLoaded', () => { if ($('chat-drawer')?.open) { toBottom(); api.focus(); } });
  return api;
})();

// <details data-keep-open>: o servidor re-renderiza o guia a cada pausa do relato; o checklist aberto continua aberto.
(function () {
  const open = new Set();
  document.addEventListener('toggle', (e) => {
    const d = e.target; if (!(d instanceof HTMLDetailsElement) || !d.hasAttribute('data-keep-open') || !d.id) return;
    if (d.open) open.add(d.id); else open.delete(d.id);
  }, true);
  const restore = () => open.forEach((id) => { const d = document.getElementById(id); if (d && !d.open) d.open = true; });
  // after:process dispara no conteúdo novo já inserido (o elemento que fez a requisição pode ter sido trocado)
  document.addEventListener('htmx:after:process', restore);
})();
