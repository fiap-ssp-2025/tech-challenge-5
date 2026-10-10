# Guardiã AI — Relatório técnico

Tech Challenge 5 · FIAP Pós-Tech IA para Devs · grupo fiap-ssp-2025
Marcelo Arruda de Siqueira · Leonardo Barbosa Nogueira · Jose Flavio Neto · Pedro Matias dos Santos · Wellington Oliveira de Andrade

Repositório: `tech-challenge-5` · Modelo publicado: `huggingface.co/fiap-ssp-2025/tc5-guardia-encaminhamento` · Vídeo: (link na entrega)

## 1. Problema e cenário

Uma instituição que atende mulheres recebe, num mesmo atendimento, relato em texto livre, dados cadastrais e respostas de
formulário. A profissional precisa identificar o que merece atenção, decidir próximos passos e consultar protocolos. O
roteiro da Fase 5 pede uma aplicação que receba essas informações e use técnicas de IA para apoiar a análise inicial,
com a decisão final sempre humana.

Escolhemos o cenário de **segurança da mulher** (violência doméstica e familiar) como cenário principal e único. A
dimensão de saúde aparece dentro da jornada (encaminhamentos de saúde, norma técnica do Ministério da Saúde sobre
violência sexual), não como um segundo fluxo. A decisão está documentada em `spec.md` (D1): dois escopos dobrariam
teste, demonstração e relatório sem ganhar nenhum requisito.

A Guardiã AI é a evolução dos quatro Tech Challenges anteriores do grupo: dados e extração do SINAN (TC1), modelo de
risco com algoritmo genético e SHAP (TC2), LangGraph, RAG, guardrails e auditoria (TC3) e a infraestrutura de GPU no
RunPod usada no TC4.

## 2. Dados

**Fonte.** SINAN/DATASUS, notificações de violência interpessoal e autoprovocada, Brasil, 2024 (dados públicos e
anonimizados; 662.803 notificações, 160 campos). Extração DBC → parquet herdada do TC1 (`pysus`).

**Filtro.** Mulheres (`CS_SEXO = F`), lesão não autoprovocada (`LES_AUTOP = 2`) e pelo menos um dos quatro campos de
encaminhamento à rede de proteção respondido (Sim ou Não). Resultado: **280.757 notificações**.

**Alvo: `ENCAMINHA_PROTECAO`.** Vale 1 quando a ficha registra encaminhamento para Delegacia da Mulher (`DELEG_MULH`),
Centro de Atendimento à Mulher (`ATEND_MULH`), Ministério Público (`MPU`) ou Defensoria Pública (`DEFEN_PUBL`).
Prevalência: **28,3%**.

**Por que não "recorrência".** Nas fases 1 e 2 o alvo era `OUT_VEZES` ("ocorreu outras vezes?"). Ao preparar esta fase
verificamos que esse campo é preenchido na mesma ficha que todas as features: não há ordem temporal, é um estudo
transversal e não uma predição. No produto o problema é ainda mais direto: a atendente pode perguntar. O SINAN não
registra desfecho futuro (`EVOLUCAO`, `CLASSI_FIN` e `DT_OBITO` estão vazios) nem tem identificador para ligar
notificações da mesma mulher, mas registra em cerca de 94% das fichas para onde o caso foi encaminhado. Esse registro é
uma decisão profissional posterior aos fatos relatados e vira um alvo de apoio à decisão honesto: *"casos com este
perfil foram encaminhados à rede de proteção em 2024?"*. `OUT_VEZES` passou a ser feature, que é o seu papel natural.

**Features (42).** 10 tipos de violência e 9 meios de agressão (binários), 5 grupos de relação com o provável autor
(parceiro íntimo, familiar, conhecido, institucional, outros), 13 campos nominais de contexto (repetição, álcool, sexo do
autor, número de envolvidos, raça/cor, ciclo de vida, situação conjugal, gestação, orientação sexual, identidade de
gênero, deficiência, local, motivação), UF, escolaridade ordinal (com indicador de ausência) e idade em anos (com
indicador de ausência). Campos em branco ou "ignorado" viram a categoria `IGNORADO`: nada é descartado por valor ausente,
ao contrário do TC1, que perdeu 81% das linhas com `dropna()`. Foram removidos dia da semana, mês e circunstância CID
(60% vazia). Os quatro campos que formam o alvo nunca entram nas features (teste automatizado garante isso).

**Pré-processamento.** `ColumnTransformer` com `OneHotEncoder(handle_unknown="ignore")` para nominais, imputação pela
mediana para numéricas e passagem direta para binárias, dentro de um `Pipeline` ajustado só no treino. A mesma função
(`row_from_case`) transforma um caso do formulário web, o que impede divergência entre treino e inferência.

## 3. Modelos de machine learning

**Protocolo.** Divisão 70/30 estratificada (semente 42). Dentro do treino, 15% viram validação para escolher o limiar,
ajustar a calibração e servir de fitness ao algoritmo genético. O conjunto de teste é tocado uma única vez.

**Modelos comparados.**
1. Regressão Logística (`class_weight="balanced"`), o baseline interpretável, com coeficientes reportados.
2. Random Forest com busca de hiperparâmetros por algoritmo genético (população 6, 4 gerações, elitismo 2, fitness =
   0,5·PR-AUC + 0,3·recall + 0,2·F1 na validação), herança do TC2.
3. HistGradientBoosting (`class_weight="balanced"`, parada antecipada).

**Métricas e por que elas.** A prevalência é 28%: "nunca sugerir" acerta 72% e não ajuda ninguém, então acurácia não
serve de critério. A métrica prioritária é o **recall da classe encaminhada**: um falso negativo deixa de levar à equipe
um caso com perfil de risco; um falso positivo custa uma conversa a mais. Reportamos precisão, F1, especificidade, PR-AUC
(a métrica de ordenação adequada a classes desbalanceadas), ROC-AUC e Brier. O limiar operacional maximiza F2 na
validação, pesando recall quatro vezes mais que precisão. O modelo implantado é calibrado por regressão isotônica, para
que a probabilidade mostrada à profissional tenha significado de frequência.

**Seleção.** O modelo implantado é o de melhor PR-AUC na validação, com uma regra prática: uma floresta pesada só vence o
boosting se o superar por mais de 0,01 de PR-AUC, porque o HistGradientBoosting cabe em ~1 MB e explica em milissegundos.

<!-- METRICS:START -->
Conjunto: 280.757 linhas · treino 167.049 · validação 29.480 · teste 84.228 · prevalência 0.283 · acurácia da classe majoritária 0.717.

| Modelo | PR-AUC | ROC-AUC | Recall@thr | Precisão@thr | F1@thr | Especificidade | Limiar |
|---|---:|---:|---:|---:|---:|---:|---:|
| logistic_regression | 0.612 | 0.777 | 0.906 | 0.375 | 0.531 | 0.406 | 0.308 |
| random_forest | 0.652 | 0.803 | 0.930 | 0.378 | 0.537 | 0.396 | 0.313 |
| hist_gradient_boosting **(implantado)** | 0.674 | 0.816 | 0.923 | 0.392 | 0.551 | 0.437 | 0.287 |
| random_forest_ga | 0.680 | 0.819 | 0.918 | 0.401 | 0.558 | 0.460 | 0.280 |

Modelo implantado: **hist_gradient_boosting** calibrado (isotônica): Brier 0.1739 → 0.1462; no limiar calibrado 0.150: recall 0.923, precisão 0.392, F1 0.551. Gap de recall por raça/cor: 0.180. Artefato: 0.5 MB. Figuras em `docs/figures/` (PR, calibração, SHAP).
Algoritmo genético (RF): melhores hiperparâmetros {'n_estimators': 150, 'max_depth': 20, 'min_samples_leaf': 10, 'max_features': 0.3}.
Top features (média |SHAP|): IDADE_ANOS, REL_PARCEIRO_INTIMO, VIOL_MOTIV_SEXISMO, AG_AMEACA, AUTOR_SEXO_MASCULINO, OUT_VEZES_SIM.
<!-- METRICS:END -->

**Explicabilidade.** SHAP (`TreeExplainer`) global sobre uma amostra do teste e por caso na interface: os seis fatores
de maior |SHAP| aparecem na nota com o sentido (aumenta ou reduz a probabilidade) e são passados à LLM, que os traduz
para a profissional. Para a regressão logística, as contribuições coeficiente × valor fazem o mesmo papel.

**Equidade.** Recall e taxa de sugestão por raça/cor no teste, com o gap de recall reportado. O modelo aprende a prática
registrada em 2024, o que inclui diferenças regionais e sub-encaminhamento; por isso a interface diz "perfil encaminhado
em 2024", nunca "deve ser encaminhada".

**Impacto dos erros.** Falso negativo: a profissional pode não considerar a rede de proteção num caso de risco; é o erro
mais caro e o motivo do limiar baixo. Falso positivo: uma discussão a mais com a equipe. Em ambos os casos a decisão é
humana e a nota traz o que ainda precisa ser perguntado.

## 4. Aplicação e orquestração (LangGraph)

A aplicação é um site renderizado no servidor (FastAPI + Jinja2) com htmx 4, Tailwind CSS 4 e Basecoat; sem SPA. O
servidor é a única fonte de verdade: cada URL responde a página completa no acesso direto e o fragmento quando chamada
pelo htmx; erros de formulário voltam como 422 com o formulário re-renderizado. Tema claro/escuro, responsivo a partir de
360 px, estados de carregando, vazio, erro e sucesso em toda interação.

**Fluxos.** F1 Triagem do relato (principal), F2 Estruturar relato, F3 Consultar protocolos, F4 Histórico e auditoria,
páginas Modelo e Status. F5 (relato por áudio, whisper do TC4) e F6 (painel do SINAN) ficaram como opcionais.

**Grafo do F1 (9 nós, `guardia/graph/flow.py`).**

```
assess_risk → run_model → retrieve_protocols → define_actions → build_explanation
→ prepare_audit → generate_note → judge → persist → END
```

- `assess_risk`: regras determinísticas sobre texto, formulário e sinais (arma, ameaça de morte, cárcere, repetição,
  gestação, filhos, local seguro) → nível baixo/moderado/alto/crítico e fatores.
- `run_model`: probabilidade calibrada, limiar, sugestão e SHAP do caso.
- `retrieve_protocols`: busca semântica na base de protocolos; guarda quais documentos contribuíram.
- `define_actions`: ações e encaminhamentos por nível de risco e sugestão do modelo (190, DEAM, medida protetiva,
  Casa da Mulher Brasileira, CREAS, Defensoria/MP, Ligue 180, saúde em violência sexual).
- `generate_note`: a LLM geradora recebe relato, ficha, risco, leitura do modelo, trechos recuperados e ações, e devolve
  a nota em JSON restrito por schema (nível, resumo, fatores, leitura do modelo, ações, encaminhamentos, perguntas
  pendentes, fontes citadas, aviso). Blocklist de frases proibidas em seguida.
- `judge`: LLM-as-judge em outro servidor e modelo, seis critérios; reprovação bloqueia a nota.
- `persist`: SQLite com hash SHA-256 do relato (texto nunca gravado), risco, probabilidade, fatores, ficha, fontes,
  backend, latência, veredito e a nota.

**F2.** A mesma LLM extrai do relato os campos da ficha (booleans, enums, idade) com schema fechado e lista os campos
incertos; a profissional confere antes de analisar. **F3.** Chat sobre a base RAG com memória no servidor, resposta
restrita aos trechos recuperados, cobertura declarada (total/parcial/nenhuma) e fontes citadas.

## 5. LLM, RAG e segurança

**Geradora.** `QuantTrio/Qwen3.6-27B-AWQ` (27B, 4 bits) servido por vLLM num pod RunPod (GPU A40/L40S 48 GB), endpoint
compatível com OpenAI, saída estruturada por `response_format=json_schema` (decodificação restrita: o modelo não consegue
gerar JSON fora do schema). Fallback opcional para OpenAI. O cliente limpa erros de proxy e registra backend, modelo,
latência e tokens em cada chamada.

**Juiz.** `Qwen/Qwen2.5-7B-Instruct-AWQ` num segundo servidor vLLM do mesmo pod. Critérios: não prescreve, não fecha
diagnóstico, não decide pela profissional, encaminha à rede de proteção quando há violência, linguagem respeitosa,
privacidade. Defesas herdadas do TC3 contra juiz incoerente: reprovar exige trecho literal presente na nota; justificativa
que nega o problema vira aprovação; padrões de posologia, afirmação diagnóstica e dado pessoal são checados por regex.

**RAG.** 11 documentos (`data/protocols`): Lei 11.340/2006 e resumo institucional, tipos de violência, medidas
protetivas, canais nacionais e do DF, rede de proteção do DF e seu protocolo, LGPD e a norma técnica MS/MJ sobre
violência sexual. Chunks de 700 caracteres, embeddings `paraphrase-multilingual-MiniLM-L12-v2`, Chroma persistente com
distância cosseno, no máximo 2 trechos por documento. A nota cita os títulos usados; a interface distingue recuperado de
citado e permite ver os trechos.

**Guardrails e responsabilidade.** Aviso fixo em toda resposta; blocklist; juiz independente; nenhum dado pessoal é
necessário nem persistido; auditoria completa de cada análise; LangSmith rastreia cada nó e chamada. O sistema não
decide encaminhamentos, não diagnostica e não prescreve.

## 6. Avaliação da aplicação

Relatos sintéticos: como o SINAN não tem texto livre, a geradora escreve relatos curtos a partir de fichas reais
(`scripts/gerar_relatos_sinteticos.py`); os rótulos vêm da própria ficha. Sobre eles, `scripts/avaliar_pipeline.py` mede:
F2, concordância por campo entre o extraído e a ficha de origem; F1, aprovação do juiz, bloqueios, latência, fontes
recuperadas e citadas, distribuição de risco e concordância entre a sugestão do modelo e o encaminhamento real.

<!-- EVAL:START -->
Relatos sintéticos avaliados: **30**.

**F2 (estruturar relato).** 30/30 extrações válidas; concordância por campo binário com a ficha de origem: precisão 0.916, recall 0.932; acerto de `OUT_VEZES` 1.000; latência mediana 21953 ms.

**F1 (grafo completo).** 30/30 notas geradas pela LLM; juiz avaliou 30 e aprovou 29; 1 bloqueadas (blocklist ou juiz). Latência mediana da LLM 24938 ms, tempo mediano total 51.6 s. Fontes recuperadas por nota: 2.8 (citadas 2.1). Concordância entre sugestão do modelo e encaminhamento real da ficha: 0.767. Distribuição de risco por regras: {'baixo': 0, 'moderado': 0, 'alto': 24, 'critico': 6}.
<!-- EVAL:END -->

## 7. Execução e entrega

`make install env data train serve` no clone; `make test` roda a suíte (features, regras do grafo, juiz, rotas web);
`make docker-build` e `make docker-run` sobem o container com modelo, protocolos e índice RAG dentro da imagem e LLM
remota por variável de ambiente; `make pod-create` provisiona os dois servidores vLLM no RunPod; `make publish-hf`
publica o modelo na organização do grupo. CI no GitHub Actions executa lint, testes e build da imagem.

## 8. Limitações

- O alvo é a prática registrada pelos serviços de saúde em 2024, não um padrão-ouro clínico ou jurídico.
- Transversal: não prevê desfecho futuro; apoia a leitura do caso presente.
- Base RAG pequena e concentrada no DF; respostas fora dela são sinalizadas como sem cobertura.
- Relatos sintéticos substituem relatos reais na avaliação; a concordância do F2 mede consistência, não verdade.
- A LLM depende de um pod alugado. Sem ele a nota degrada para regras + modelo, com a falha registrada na auditoria.
- O juiz reduz, mas não elimina, respostas inadequadas; a decisão é da profissional.

## 9. Próximos passos

Classificador local e rápido (Laya) para arma/relação/repetição comparado ao F2; relato por áudio com faster-whisper
(F5); ampliar a base de protocolos com normas estaduais; avaliação com profissionais da rede.
