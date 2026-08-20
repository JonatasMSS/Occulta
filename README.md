# Anonimização seletiva de rostos em vídeo

Esta aplicação mantém visível a pessoa presente na imagem de referência e desfoca todos os outros rostos detectados no vídeo. Ela processa cada frame, produz um MP4 H.264 e preserva o áudio de origem quando existir.

## Como funciona

~~~
Imagem de referência -> detectar um rosto -> embedding ArcFace

Vídeo -> para cada frame:
        detectar rostos periodicamente (RetinaFace)
        rastrear caixas entre detecções (OpenCV MIL)
         -> comparar com a referência (similaridade de cosseno)
         -> manter o alvo / desfocar não-alvos
         -> escrever frame
      -> FFmpeg: H.264 + áudio original (ou AAC)
~~~

A Chain of Responsibility de execução é:

~~~
Validar entradas -> Preparar referência -> Processar vídeo -> Exportar MP4
~~~

Durante o processamento de vídeo, cada frame percorre a cadeia:

~~~
Detectar faces -> Identificar alvo -> Desfocar não-alvos -> Gravar frame
~~~

## Requisitos

- Python 3.13 ou superior;
- FFmpeg disponível no PATH;
- dependências declaradas em pyproject.toml;
- conexão com a internet na primeira execução, para baixar os pesos ArcFace do DeepFace.

O DeepFace usa GPU automaticamente quando o ambiente compatível estiver disponível; caso contrário, usa CPU. No Windows nativo com TensorFlow 2.11 ou superior, CUDA não é suportado: use WSL2 com GPU NVIDIA configurada para acelerar a inferência.

O projeto seleciona automaticamente tf-keras (Keras 2) antes de carregar TensorFlow, pois o RetinaFace ainda não é compatível com a API Keras 3.

## Instalação

~~~
python -m venv .venv
.venv\Scripts\activate
pip install -e .
ffmpeg -version
~~~

## Uso

~~~
python main.py ^
  --reference-image data\reference\pessoa.jpg ^
  --input-video data\input\video.mp4 ^
  --output-video data\output\anonimizado.mp4 ^
  --similarity-threshold 0.80 ^
  --detection-interval 5
~~~

Opcionalmente, use --debug-dir debug para salvar a referência anotada e o primeiro frame com faces, caixas e pontuações.

--similarity-threshold aceita valores entre 0 e 1. A similaridade é a cosseno entre embeddings ArcFace: faces com valor maior ou igual ao limiar são consideradas o alvo e permanecem nítidas. O padrão é 0.80.

--detection-interval define a quantidade de frames entre detecções RetinaFace completas. O padrão 5 usa rastreamento OpenCV MIL nos frames intermediários e gera embeddings ArcFace em lote nos frames detectados, reduzindo o custo. Use 1 para detectar em todos os frames quando a prioridade for precisão máxima.

Durante a execução, tqdm mostra o andamento dos frames, velocidade, estimativa de tempo e os contadores de rostos mantidos e desfocados. As quatro etapas também são exibidas antes de iniciar.

O caminho de saída deve terminar em .mp4. Caso o arquivo já exista, ele é substituído somente após uma exportação bem-sucedida. Vídeos locais, modelos e resultados gerados são ignorados pelo Git.

## Limitações

- A imagem de referência deve conter exatamente um rosto.
- A V1 redetecta periodicamente e usa rastreamento entre detecções; novos rostos podem aparecer por até detection-interval menos um frames antes da próxima detecção.
- Um rosto que o detector não encontrar não pode ser desfocado.
- Iluminação baixa, oclusões, rostos pequenos ou parecidos podem exigir ajustar o limiar.
- Revise o vídeo final antes de compartilhar conteúdo que exija proteção de identidade.
