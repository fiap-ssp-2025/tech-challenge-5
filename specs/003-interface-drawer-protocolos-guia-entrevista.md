# Spec 003 — Interface: protocolos em drawer e guia de perguntas ao vivo

Status: **done** (2026-10-10) · Base: `spec.md` v1.1, `AGENTS.md`, spec 002 (done).

## Objetivo

Duas mudanças de interface pedidas pelo grupo, sem mexer na arquitetura (HTML no servidor, htmx 4, Basecoat):

1. **Protocolos (F3) saem da navegação e viram um painel lateral** aberto por um botão fixo no canto, no estilo "suporte" de site. A conversa é contínua (balões, memória no servidor) e pode ser recomeçada.
2. **A tela de triagem ganha um guia de entrevista:** enquanto o relato é falado ou digitado, a Guardiã mostra, de forma mínima, as próximas perguntas que ainda faltam fazer à mulher.

## Decisões

| # | Decisão | Motivo |
|---|---|---|
| U1 | **Drawer = `<dialog>` nativo aberto com `show()` (não modal)**, à direita, largura `min(26rem, 100vw)`. Em telas ≥ 1100 px o conteúdo da página encolhe para o lado (o relato continua visível e editável); abaixo disso o painel cobre a tela. Esc e o botão × fecham; o foco volta ao botão flutuante. | A profissional consulta um protocolo no meio do atendimento sem perder o relato. Modal travaria a página; `<dialog>` nativo dá semântica e foco sem biblioteca (AGENTS: nativo antes de JS). |
| U2 | **Botão flutuante "Protocolos"** no canto inferior direito, em todas as páginas; a conversa carrega na primeira abertura (`hx-trigger="click once"`). `/protocolos` deixa de ser página: acesso direto abre a triagem com o painel aberto; com `HX-Request` devolve só o fio da conversa. | Mesma URL, página completa ou fragmento (regra do `render`). Link antigo continua funcionando. |
| U3 | **Conversa contínua:** balões (você à direita, Guardiã à esquerda), fontes recolhidas sob cada resposta, Enter envia e Shift+Enter quebra linha, a pergunta aparece na hora (balão provisório + "consultando os protocolos…") e a resposta do servidor o substitui. **Recomeçar** cria outro id de conversa (cookie) e volta ao estado vazio com sugestões. | Pedido do grupo: conversa, não formulário de pergunta e resposta. A memória já existia no servidor (6 turnos). |
| U4 | **Aviso de segurança fixo no rodapé do painel**, não repetido em cada balão. O texto continua gravado com a resposta. | Numa conversa longa o aviso repetido vira ruído; continua sempre visível. |
| U5 | **Guia de entrevista = checklist de 11 tópicos** inspirado no Formulário Nacional de Avaliação de Risco (autor, o que aconteceu, ameaça de morte, arma, recorrência, local seguro, filhos, gestação, álcool/drogas, medida protetiva, idade), em ordem de prioridade de risco. Mostra só as **3 próximas perguntas**, escritas para ler em voz alta, e um contador "n de 11" que expande o checklist inteiro. | Pedido do grupo: diretrizes "bem minimalistas". Perguntas fixas e revisáveis são previsíveis e auditáveis; a LLM não inventa perguntas. |
| U6 | **Um tópico conta como abordado** se a ficha já tem o campo (preenchido pela LLM ou à mão) **ou** se a extração ao vivo diz que o relato tocou no assunto, mesmo com resposta negativa ("ele não tem arma"). A extração ganha `topicos_abordados` nos dois schemas, cada item como `"id: trecho literal"`, e o servidor só aceita o tópico se o trecho estiver no relato (ver spec 004, C6). A profissional também pode marcar "já perguntei" clicando na pergunta. | Ficha só sabe "sim"; o "não" dito no relato também fecha a pergunta. Lista de ids custa poucos tokens. |
| U7 | **O guia é renderizado no servidor** (`partials/guia.html`): vem como OOB junto da ficha em `/estruturar`, e `POST /guia` o recalcula a partir do formulário quando a profissional edita a ficha ou marca uma pergunta. Com a ficha editada à mão, a extração ao vivo continua rodando, mas só atualiza o guia. | Servidor como fonte única; o estado (tópicos vistos pela LLM, marcados à mão) vive em campos ocultos do próprio formulário. |
| U9 | **Pergunta de seguimento busca com a anterior:** a consulta ao RAG junta a última pergunta do fio à atual. | Medido no teste: "e quanto tempo o juiz tem para decidir?" sozinha recuperava a norma de violência sexual; com a anterior vêm os documentos de medidas protetivas. (O prazo de 48 h não está em nenhum dos 11 documentos; a resposta "sem trecho que a sustente" estava correta.) |
| U8 | Requisições com `HX-Request-Type: full` (navegação com `hx-boost` para o `body`) recebem a página inteira. | A navegação reforçada do cabeçalho recebia o fragmento da página. |

## Tarefas

- [x] T1 `guardia/interview.py`: tópicos (id, rótulo, pergunta, cobertura pela ficha) e `build_guide()`.
- [x] T2 `llm/prompts.py` + `llm/structurer.py`: campo `topicos_abordados` nos schemas completo e esparso; `Structured.topics`.
- [x] T3 `web/app.py`: `/estruturar` devolve ficha + guia OOB (ou só o guia com a ficha editada); `POST /guia`; `/protocolos` vira fragmento/página com painel aberto; `/protocolos/nova` devolve o fio vazio; `is_htmx` respeita `HX-Request-Type: full`; sai `Protocolos` do `NAV`.
- [x] T4 Templates: `partials/guia.html`, `partials/chat_drawer.html`, `partials/chat_thread.html`; `chat_turn.html` em balões; triagem com o guia sob a caixa do relato; `pages/protocolos.html` removido.
- [x] T5 `static/app.js` (abrir/fechar painel, foco, Esc, envio com Enter, balão provisório, manter o checklist aberto entre atualizações) e `static/app.css` (drawer, botão flutuante, balões, guia; claro/escuro; movimento reduzido).
- [x] T6 Testes: `/guia`, `/protocolos` (fragmento × página), OOB do guia, `build_guide`.
- [x] T7 Docs: `spec.md` (D19–D20, §10), README (F3), `docs/roteiro_video.md`, AGENTS.md.

## Critérios de aceite (verificados em 2026-10-10 no navegador, 1440 px claro, 1024 px escuro e 375 px)

- Botão "Protocolos" visível em `/`, `/historico` e no detalhe; abre o painel com foco no campo de pergunta; Esc fecha e devolve o foco.
- Pergunta aparece imediatamente; resposta chega no mesmo fio com fontes recolhidas; segunda pergunta usa a memória; "Recomeçar" limpa o fio.
- Em ≥ 1100 px, painel aberto e relato editável lado a lado, sem scroll horizontal; em 360 px o painel ocupa a tela.
- Na triagem vazia o guia mostra 3 perguntas iniciais; ao ditar "ele ameaçou me matar com uma faca, já aconteceu outras vezes", ameaça, arma e recorrência saem da lista; clicar numa pergunta a marca como feita; o contador abre o checklist inteiro e ele não fecha sozinho a cada atualização.
- `make test` verde, `ruff` limpo, tema claro e escuro conferidos.

## Fora de escopo

O chat de protocolos usar o caso em andamento como contexto (próximo passo natural), streaming token a token da resposta, guia gerado por LLM.
