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
