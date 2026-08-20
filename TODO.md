# Sistema de anonimização seletiva em vídeos — TODO

> Objetivo da V1: receber uma imagem de referência e um vídeo, localizar **uma única pessoa** no vídeo e desfocar somente o rosto dela, preservando o áudio.

## Como usar este arquivo

- Marque cada item concluído com `- [x]`.
- Preencha o campo **Aprendi** ao fim de cada etapa.
- Não avance se o respectivo critério de aceite ainda não estiver atendido.
- Registre problemas e decisões: eles serão úteis na avaliação e na documentação final.

---

## 0. Escopo e decisões iniciais

- [x] Criar o repositório Git e o repositório remoto no GitHub.
- [x] Adicionar um `.gitignore` para Python, ambientes virtuais, vídeos, modelos e arquivos temporários.
- [x] Criar um `README.md` com objetivo, escopo da V1 e instruções de execução vazias/iniciais.
- [ ] Definir que a V1 identifica apenas uma pessoa por execução.
- [ ] Definir entradas: `imagem_referencia`, `video_entrada`, `video_saida` e limiar de similaridade.
- [ ] Definir saída esperada: vídeo MP4 processado com áudio preservado.
- [ ] Definir limitações explícitas: oclusão, ângulos extremos, baixa luz, vários rostos similares e falhas de detecção.
- [ ] Escolher a stack de embeddings: PyTorch **ou** TensorFlow (escolher uma só para a V1).

**Critério de aceite:** escopo e interface de entrada/saída estão escritos no README.

**Aprendi:**

> _Preencha aqui._

**Decisões/problemas:**

> _Preencha aqui._

---

## 1. Preparar o ambiente

- [ ] Instalar Python compatível com a biblioteca/modelo escolhido.
- [ ] Criar ambiente virtual (`.venv`).
- [ ] Criar `requirements.txt` com dependências mínimas.
- [ ] Instalar OpenCV, NumPy, RetinaFace e a implementação escolhida de FaceNet.
- [ ] Instalar/configurar FFmpeg e confirmar que `ffmpeg -version` funciona no terminal.
- [ ] Criar a estrutura inicial:

  ```text
  src/
  tests/
  data/reference/
  data/input/
  data/output/
  docs/
  ```

- [ ] Adicionar um script mínimo que importe todas as dependências.
- [ ] Executar o script e corrigir erros de instalação/modelos.

**Critério de aceite:** ambiente reproduzível e imports funcionando em uma máquina limpa/ambiente virtual.

**Aprendi:**

> _Preencha aqui._

**Decisões/problemas:**

> _Preencha aqui._

---

## 2. Organizar dados de teste

- [ ] Separar pelo menos uma imagem de referência com um rosto nítido.
- [ ] Separar um vídeo curto contendo a pessoa-alvo.
- [ ] Separar um vídeo curto sem a pessoa-alvo (caso negativo).
- [ ] Separar, se possível, um vídeo com mais de um rosto.
- [ ] Garantir que os arquivos de vídeo reais não sejam enviados ao Git.
- [ ] Documentar formato, resolução, FPS e duração dos vídeos de teste.
- [ ] Verificar se os vídeos podem ser lidos pelo OpenCV.
- [ ] Verificar se há áudio nos vídeos com FFmpeg/ffprobe.

**Critério de aceite:** há dados positivos e negativos suficientes para testar a identificação seletiva.

**Aprendi:**

> _Preencha aqui._

**Decisões/problemas:**

> _Preencha aqui._

---

## 3. Detectar rosto na imagem de referência

- [ ] Criar o módulo de detecção de rostos.
- [ ] Carregar a imagem de referência.
- [ ] Executar RetinaFace na imagem.
- [ ] Exibir/logar quantidade de rostos detectados.
- [ ] Validar que existe exatamente um rosto na imagem de referência.
- [ ] Tratar os erros: imagem inexistente, ilegível, nenhum rosto e mais de um rosto.
- [ ] Extrair a caixa delimitadora e os landmarks do rosto detectado.
- [ ] Salvar uma imagem de depuração com caixa e landmarks desenhados.
- [ ] Conferir visualmente se a caixa cobre corretamente o rosto.

**Critério de aceite:** uma imagem com uma pessoa gera uma única caixa facial correta; entradas inválidas falham com mensagem clara.

**Aprendi:**

> _Preencha aqui._

**Decisões/problemas:**

> _Preencha aqui._

---

## 4. Preparar o rosto para o FaceNet

- [ ] Definir se será usado alinhamento facial pelos landmarks.
- [ ] Implementar o recorte da região facial a partir da caixa detectada.
- [ ] Garantir que coordenadas fora da imagem sejam limitadas corretamente.
- [ ] Redimensionar o recorte para o tamanho exigido pelo FaceNet escolhido.
- [ ] Converter BGR (OpenCV) para RGB quando necessário.
- [ ] Aplicar a normalização exigida pelo modelo.
- [ ] Salvar/visualizar o rosto pré-processado para depuração.
- [ ] Conferir que a entrada final tem forma e tipo corretos.

**Critério de aceite:** todo recorte facial válido se torna uma entrada compatível com o FaceNet.

**Aprendi:**

> _Preencha aqui._

**Decisões/problemas:**

> _Preencha aqui._

---

## 5. Gerar e validar o embedding de referência

- [ ] Carregar o modelo FaceNet pré-treinado uma única vez.
- [ ] Gerar o embedding do rosto de referência.
- [ ] Registrar a dimensão do vetor gerado.
- [ ] Normalizar o embedding se a implementação escolhida exigir/recomendar isso.
- [ ] Salvar o embedding em memória para a V1.
- [ ] Testar duas execuções sobre a mesma imagem e confirmar embeddings consistentes.
- [ ] Testar outra imagem da mesma pessoa, se disponível.
- [ ] Testar uma imagem de pessoa diferente, se disponível.

**Critério de aceite:** o embedding é gerado de forma estável e pode distinguir, ao menos nos testes simples, mesma pessoa de pessoa diferente.

**Aprendi:**

> _Preencha aqui._

**Decisões/problemas:**

> _Preencha aqui._

---

## 6. Comparar embeddings e escolher o limiar

- [ ] Escolher uma métrica: distância euclidiana ou similaridade do cosseno.
- [ ] Implementar uma função pequena de comparação.
- [ ] Definir o sentido da decisão: menor distância ou maior similaridade.
- [ ] Medir valores para pares da mesma pessoa.
- [ ] Medir valores para pares de pessoas diferentes.
- [ ] Escolher um limiar inicial baseado nas medições.
- [ ] Expor o limiar como argumento de linha de comando/configuração simples.
- [ ] Registrar o valor inicial e a justificativa no README.
- [ ] Testar caso exatamente no limiar.

**Critério de aceite:** a decisão “é a pessoa-alvo?” funciona sobre exemplos positivos e negativos conhecidos.

**Aprendi:**

> _Preencha aqui._

**Decisões/problemas:**

> _Preencha aqui._

---

## 7. Detectar rostos em um frame de vídeo

- [ ] Abrir o vídeo com `cv2.VideoCapture`.
- [ ] Ler metadados: largura, altura, FPS e quantidade de frames (quando disponível).
- [ ] Ler apenas o primeiro frame válido.
- [ ] Executar RetinaFace no frame.
- [ ] Iterar sobre todos os rostos detectados.
- [ ] Desenhar caixas, landmarks e índices dos rostos para depuração.
- [ ] Salvar o frame anotado.
- [ ] Tratar vídeo vazio, corrompido ou sem frames.

**Critério de aceite:** o primeiro frame gera uma visualização correta de todos os rostos detectados.

**Aprendi:**

> _Preencha aqui._

**Decisões/problemas:**

> _Preencha aqui._

---

## 8. Identificar a pessoa-alvo em um frame

- [ ] Para cada rosto detectado, recortar e pré-processar a face.
- [ ] Gerar o embedding de cada rosto do frame.
- [ ] Comparar cada embedding com o embedding de referência.
- [ ] Marcar somente os rostos que passam pelo limiar.
- [ ] Desenhar no frame de depuração a pontuação de cada rosto.
- [ ] Verificar cenário com apenas a pessoa-alvo.
- [ ] Verificar cenário com outra pessoa, sem alvo.
- [ ] Verificar cenário com alvo e outra pessoa no mesmo frame.

**Critério de aceite:** somente o rosto correspondente à referência é marcado no frame de teste misto.

**Aprendi:**

> _Preencha aqui._

**Decisões/problemas:**

> _Preencha aqui._

---

## 9. Aplicar o desfoque facial

- [ ] Criar uma função que receba frame e caixa facial.
- [ ] Ajustar/validar os limites da caixa antes de recortar.
- [ ] Aplicar `cv2.GaussianBlur` somente na região do rosto.
- [ ] Escolher um tamanho de kernel proporcional ou adequado à resolução do vídeo.
- [ ] Testar em uma imagem/frame estático.
- [ ] Confirmar visualmente que a identidade fica ocultada.
- [ ] Confirmar que áreas fora da caixa não foram alteradas.
- [ ] Testar caixas próximas às bordas do frame.

**Critério de aceite:** o rosto-alvo fica irreconhecível e o restante do frame permanece intacto.

**Aprendi:**

> _Preencha aqui._

**Decisões/problemas:**

> _Preencha aqui._

---

## 10. Processar o vídeo completo, sem áudio

- [ ] Criar o loop principal de leitura de frames.
- [ ] Carregar detector e FaceNet antes do loop.
- [ ] Para cada frame: detectar, gerar embeddings, comparar e desfocar o alvo.
- [ ] Criar `cv2.VideoWriter` com resolução e FPS do vídeo original.
- [ ] Escrever cada frame processado em um vídeo temporário sem áudio.
- [ ] Exibir progresso simples (frame atual/total ou percentual).
- [ ] Liberar `VideoCapture` e `VideoWriter`, inclusive em caso de erro.
- [ ] Processar um vídeo curto de ponta a ponta.
- [ ] Reproduzir o vídeo temporário e conferir sincronização visual.

**Critério de aceite:** o vídeo processado contém todos os frames e desfoca somente a pessoa-alvo, mas ainda pode estar sem áudio.

**Aprendi:**

> _Preencha aqui._

**Decisões/problemas:**

> _Preencha aqui._

---

## 11. Preservar/recombinar o áudio com FFmpeg

- [ ] Confirmar que o vídeo de entrada possui faixa de áudio.
- [ ] Criar comando FFmpeg para combinar vídeo processado e áudio original.
- [ ] Copiar o áudio sem recodificação quando compatível.
- [ ] Garantir que o vídeo final tenha duração correta.
- [ ] Testar vídeo de entrada sem áudio e tratá-lo sem falha.
- [ ] Confirmar que o arquivo temporário é separado do arquivo final.
- [ ] Manter/remover o temporário apenas após confirmar que o arquivo final foi criado.
- [ ] Reproduzir o resultado com som e imagem.

**Critério de aceite:** o MP4 final reproduz vídeo anonimizado e áudio original sincronizado.

**Aprendi:**

> _Preencha aqui._

**Decisões/problemas:**

> _Preencha aqui._

---

## 12. Criar a interface de linha de comando

- [ ] Criar um ponto de entrada, por exemplo `python -m src.main`.
- [ ] Receber caminho da imagem de referência.
- [ ] Receber caminho do vídeo de entrada.
- [ ] Receber caminho do vídeo de saída.
- [ ] Receber limiar opcional.
- [ ] Validar caminhos antes de iniciar o processamento.
- [ ] Mostrar mensagens de erro acionáveis.
- [ ] Mostrar resumo no fim: frames processados, faces identificadas e arquivo gerado.
- [ ] Documentar um comando de uso no README.

**Critério de aceite:** uma pessoa consegue executar o projeto com um único comando documentado.

**Aprendi:**

> _Preencha aqui._

**Decisões/problemas:**

> _Preencha aqui._

---

## 13. Testar casos essenciais

- [ ] Pessoa-alvo sozinha, bem iluminada.
- [ ] Pessoa-alvo com outras pessoas no frame.
- [ ] Vídeo sem a pessoa-alvo.
- [ ] Pessoa-alvo em movimento.
- [ ] Mudança de distância para a câmera.
- [ ] Rosto parcialmente ocluído.
- [ ] Rosto pequeno no frame.
- [ ] Baixa iluminação.
- [ ] Frame sem rostos.
- [ ] Imagem de referência sem rosto.
- [ ] Imagem de referência com múltiplos rostos.
- [ ] Vídeo sem áudio.
- [ ] Vídeo de formato/caminho inválido.

Para cada caso, registre resultado, limiar utilizado e observações:

| Caso | Resultado esperado | Resultado obtido | Limiar | Observações |
| --- | --- | --- | --- | --- |
| _Ex.: alvo + outra pessoa_ | somente alvo desfocado |  |  |  |

**Critério de aceite:** casos positivos não expõem o alvo e casos negativos não desfocam pessoas indevidas nos vídeos de teste.

**Aprendi:**

> _Preencha aqui._

**Decisões/problemas:**

> _Preencha aqui._

---

## 14. Avaliar qualidade e desempenho

- [ ] Medir tempo total de processamento de um vídeo conhecido.
- [ ] Calcular FPS efetivo de processamento.
- [ ] Registrar CPU/GPU usada.
- [ ] Registrar resolução do vídeo e quantidade de frames.
- [ ] Contar falsos positivos (outras pessoas desfocadas).
- [ ] Contar falsos negativos (alvo não desfocado).
- [ ] Ajustar o limiar e repetir os testes necessários.
- [ ] Escolher a melhor configuração para a demonstração.
- [ ] Documentar limitações observadas e causas prováveis.

| Vídeo | Resolução | Frames | Tempo | FPS efetivo | Falsos positivos | Falsos negativos |
| --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |

**Critério de aceite:** há uma medição simples, reproduzível e documentada da qualidade e do desempenho da V1.

**Aprendi:**

> _Preencha aqui._

**Decisões/problemas:**

> _Preencha aqui._

---

## 15. Documentar e preparar a entrega

- [ ] Atualizar o README com requisitos, instalação e execução.
- [ ] Documentar a arquitetura: RetinaFace → pré-processamento → FaceNet → comparação → blur → FFmpeg.
- [ ] Adicionar um diagrama simples do fluxo.
- [ ] Documentar o limiar escolhido e como alterá-lo.
- [ ] Listar limitações e considerações de privacidade/uso responsável.
- [ ] Adicionar exemplos de entrada e saída sem expor dados sensíveis.
- [ ] Adicionar imagens/GIFs de demonstração, se apropriado.
- [ ] Revisar nomes de arquivos, mensagens de erro e comentários.
- [ ] Confirmar que nenhum vídeo, modelo grande, chave ou dado pessoal foi enviado ao Git.
- [ ] Fazer uma execução completa seguindo apenas o README.
- [ ] Criar uma tag/release da V1 no GitHub.

**Critério de aceite:** outra pessoa consegue instalar, executar e entender as limitações da V1 apenas com o repositório.

**Aprendi:**

> _Preencha aqui._

**Decisões/problemas:**

> _Preencha aqui._

---

## Critérios de avaliação da V1

- [ ] A imagem de referência é validada e gera um embedding facial.
- [ ] Rostos são detectados nos frames do vídeo com RetinaFace.
- [ ] Apenas a identidade de referência é escolhida para anonimização.
- [ ] O desfoque é aplicado somente ao rosto identificado.
- [ ] O vídeo final pode ser reproduzido.
- [ ] O áudio original é preservado quando existir.
- [ ] Entradas inválidas apresentam erros claros, sem gerar saída enganosa.
- [ ] README permite reproduzir a demonstração.
- [ ] Limitações, testes e aprendizados estão registrados.

## Possíveis evoluções — fora da V1

- [ ] Aceitar várias imagens de referência da mesma pessoa.
- [ ] Anonimizar várias identidades na mesma execução.
- [ ] Adicionar rastreamento facial entre frames para desempenho/estabilidade.
- [ ] Suavizar a decisão entre frames para reduzir oscilações de identificação.
- [ ] Criar interface gráfica ou web.
- [ ] Processar vídeo em tempo real.
- [ ] Oferecer outros métodos de anonimização (pixelização, máscara, troca de rosto).
- [ ] Criar métricas automatizadas e conjunto de testes maior.
