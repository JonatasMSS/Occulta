# V1 — checklist de entrega

## Implementado

- [x] CLI com imagem de referência, vídeo de entrada, vídeo MP4 de saída, limiar e diretório de depuração.
- [x] Validação de arquivos, limiar, FFmpeg e referência com exatamente um rosto.
- [x] RetinaFace para detectar rostos e DeepFace/ArcFace para embeddings.
- [x] Chain of Responsibility por frame: detectar/rastrear, comparar em lote, desfocar não-alvos e gravar.
- [x] Blur gaussiano proporcional ao rosto.
- [x] MP4 H.264 com cópia de áudio quando possível e fallback para AAC.
- [x] Limpeza de temporários e prevenção de saída parcial.
- [x] Progresso, resumo e depuração visual opcional.
- [x] GPU automática quando TensorFlow a detecta, com fallback para CPU.

## Validação manual pendente

- [ ] Alvo e outra pessoa no mesmo frame: somente o alvo fica nítido.
- [ ] Vídeo sem o alvo: todas as faces detectadas são desfocadas.
- [ ] Vídeo sem áudio: saída é gerada sem falha.
- [ ] Referência sem rosto ou com vários rostos: erro claro.
- [ ] Vídeo inválido ou vazio: erro claro e nenhuma saída parcial.
- [ ] Execução com --debug-dir: referência e primeiro frame anotados.

## Limitações conhecidas

- A V1 processa todos os frames e não usa rastreamento facial.
- Rostos não detectados não podem ser desfocados.
- O limiar padrão 0.80 deve ser calibrado com os vídeos reais.
- Vídeos de validação permanecem locais e não entram no Git.
