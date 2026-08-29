# Face Blur

Anonimização seletiva de rostos em vídeo com detector Buffalo e embeddings
AdaFace IR-101. A CLI e a interface Streamlit usam a mesma pipeline.

## Pré-requisitos

- Checkpoint em `models/adaface_ir101/model.pt`.
- FFmpeg no `PATH`.

O detector Buffalo é baixado pelo InsightFace no primeiro uso.

## Interface Streamlit local

```powershell
uv sync --extra ui
uv run streamlit run streamlit_app.py
```

Abra `http://localhost:8501`, envie um MP4, navegue até um frame e clique nas
caixas faciais. É possível acumular até 32 referências em frames diferentes.

## CUDA

O AdaFace usa CUDA automaticamente quando `torch.cuda.is_available()` retorna
`True`. O detector também usa CUDA automaticamente quando o ONNX Runtime expõe
`CUDAExecutionProvider`.

Por padrão, o `uv sync` instala o ONNX Runtime CPU. Para habilitar CUDA também
no detector, substitua esse pacote no ambiente:

```powershell
uv sync --extra ui
uv pip uninstall onnxruntime
uv pip install onnxruntime-gpu==1.29.0
uv run --no-sync streamlit run streamlit_app.py
```

Sem `CUDAExecutionProvider`, somente o detector usa CPU; o AdaFace ainda usa
CUDA se ela estiver disponível no PyTorch. A interface informa o dispositivo
usado pelo AdaFace e o provider usado pelo detector ao lado dos dados do vídeo.

## CLI

Target único:

```powershell
uv run face-blur --video video.mp4 --target target.jpg --output outputs/out.mp4
```

Múltiplos targets:

```powershell
uv run face-blur --video video.mp4 --targets-dir targets --output outputs/out.mp4
```

Use `--blur-target` para borrar os targets reconhecidos e manter as demais
faces. Sem a flag, os targets são mantidos e as demais faces são borradas.

## Testes

```powershell
uv run --extra ui python -m unittest discover -s tests -v
```
