# Giấy phép thư viện, model và dữ liệu

Mỗi thư viện, model, dataset mới phải có một dòng ở đây trước khi được dùng.
Cột "Thương mại" quyết định thứ đó có được vào bản sản phẩm hay không.

| Tên | Loại | Giấy phép | Thương mại | Dùng ở đâu |
| --- | --- | --- | --- | --- |
| PyYAML | thư viện | MIT | được | đọc luật G2P (`text/rules/*.yaml`) |
| numpy | thư viện | BSD-3-Clause | được | xử lý audio, chỉ số đo |
| soundfile (libsndfile) | thư viện | BSD-3-Clause (libsndfile: LGPL-2.1, liên kết động) | được | đọc, ghi WAV |
| pyloudnorm | thư viện | MIT | được | đo loudness ITU-R BS.1770 |
| scipy | thư viện | BSD-3-Clause | được | kéo theo pyloudnorm |
| pydantic | thư viện | MIT | được | schema manifest, speakers.json |
| pytest | công cụ dev | MIT | được | test |
| hypothesis | công cụ dev | MPL-2.0 | được (chỉ dùng khi test, không phân phối) | test chuỗi ngẫu nhiên cho chuẩn hoá |
| ruff | công cụ dev | MIT | được | lint, format |
| pyright | công cụ dev | MIT | được | kiểm tra kiểu |
| types-PyYAML | công cụ dev | Apache-2.0 | được | kiểu cho PyYAML |
| matplotlib | công cụ nghiên cứu (nhóm `viz`) | Matplotlib License (dựa trên PSF, kiểu BSD) | được | vẽ biểu đồ luận văn (`eval/plots.py`) |
| seaborn | công cụ nghiên cứu (nhóm `viz`) | BSD-3-Clause | được | biểu đồ cột đối chứng, khoảng tin cậy bootstrap |
| pandas | công cụ nghiên cứu (nhóm `viz`) | BSD-3-Clause | được | đọc log loss, bảng điểm |
| contourpy, cycler, kiwisolver, python-dateutil | thư viện (kéo theo matplotlib/pandas) | BSD-3-Clause | được | kéo theo |
| fonttools, pyparsing, pillow, tzdata, packaging | thư viện (kéo theo matplotlib/pandas) | MIT, MIT, MIT-CMU, Apache-2.0, Apache-2.0/BSD-2 | được | kéo theo |
| VieNeu v3 Turbo | model | Apache-2.0 | được, nhưng chỉ dùng làm baseline | script đánh giá riêng, không nằm trong sản phẩm |
| CosyVoice (FunAudioLLM, commit 074ca6d) | mã nguồn | Apache-2.0 (đã đọc file LICENSE gốc) | được | base model P7, nằm ở `third_party/`, không commit |
| Matcha-TTS (submodule của CosyVoice) | mã nguồn | MIT | được | kéo theo CosyVoice |
| Fun-CosyVoice3-0.5B-2512 | model | Apache-2.0 (model card); chưa có tiếng Việt | được | base model P7, nằm ở `models/`, không commit |
| Google FLEURS `vi_vn` | dataset | CC-BY-4.0 | được, phải ghi nguồn | P7 giai đoạn 1 |
| Google FLEURS-R `vi_vn` (google/fleurs-r, FLEURS phục hồi bằng Miipher, 24 kHz) | dataset | CC-BY-4.0 (thẻ dữ liệu gốc của Google) | được, phải ghi nguồn; không dựng giọng sản phẩm có tên từ người nói trong đó (CC không cấp quyền nhân thân) | `data/fleurs_r/`, thay FLEURS 16 kHz cho tầng flow 24 kHz (tải 2026-11-22) |
| huggingface_hub, hf_xet | công cụ train | Apache-2.0 | được | chỉ cài trên máy Vast.ai (`infra/vast/`), đẩy/kéo dữ liệu và checkpoint |
| YODAS vi000 (espnet/yodas) | dataset | CC-BY-3.0 (video YouTube gắn CC) | được, phải ghi nguồn từng video (`yodas_vi/SOURCES.md`); rủi ro người đăng gắn nhãn CC sai | P7 giai đoạn 1, sau khi lọc nghiêm (`infra/vast/yodas.py`) |
| Whisper large-v3-turbo (openai) | model, công cụ lọc dữ liệu | MIT | được | kiểm tra phụ đề YODAS khớp lời, không vào sản phẩm |
| PhoWhisper-small (vinai) | model, kiểm tra lúc sinh | BSD-3-Clause | được, phải ghi nguồn + giữ thông báo bản quyền | `eval_tts.py --verify`: nghe lại từng mẩu TTS, sai thì đọc lại; nếu đưa vào sản phẩm thì ghi nguồn |
| LSVSC (doof-ferb/LSVSC, Electronics 2024 13(5):977) | dataset, câu mẫu giọng | CC-BY-4.0 (bản mirror); audio gốc lấy từ Internet, chủ yếu tin tức | KHÔNG cho sản phẩm: người nói chưa đồng ý nhân bản giọng | `pick_lsvsc.py`, `pick_sweep.py` (vòng 2A): chỉ thử nội bộ model có giữ giọng miền không |
| LSVSC train (doof-ferb/LSVSC) | dataset train model gốc | CC-BY-4.0 (bản mirror), phải ghi nguồn; audio gốc từ Internet, cùng nhóm rủi ro với YODAS | người dùng đồng ý dùng để TRAIN (2026-11-20); không nhân bản giọng người trong đó cho sản phẩm | `prepare_lsvsc.py`, vòng E: ~80 h giọng Bắc/Trung/Nam nam-nữ, lọc Whisper |
| AST AudioSet (MIT/ast-finetuned-audioset-10-10-0.4593) | model, công cụ lọc dữ liệu | BSD-3-Clause | được | phát hiện nhạc/lời trong YODAS, không vào sản phẩm |
| torchaudio SQUIM objective | model, công cụ lọc dữ liệu | BSD-2-Clause (torchaudio) | được | chấm chất lượng âm thanh YODAS, không vào sản phẩm |
| Common Voice tiếng Việt | dataset | CC0 | được | P7 giai đoạn 1 (chủ dự án tự tải, cần đồng ý điều khoản của Mozilla) |
| Câu mẫu giọng "Thanh Thanh" và "Thành" | câu mẫu giọng trong gói phát hành | người nói đã đồng ý cho đăng giọng công khai dưới tên mới | miễn phí, phi thương mại; chỉ dùng tên mới | gói giọng của `vietvoice download` |
| Nhóm `tts` trong `pyproject.toml` (torch, torchaudio, transformers, onnxruntime, HyperPyYAML, modelscope, openai-whisper, conformer, diffusers, x-transformers, lightning, hydra-core, omegaconf, librosa, inflect, pyworld, pyarrow, rich, gdown, wget, tqdm, setuptools) | thư viện CosyVoice cần lúc đọc | BSD-3-Clause (torch, torchaudio), Apache-2.0 (transformers, diffusers, lightning, modelscope, HyperPyYAML, pyarrow), MIT (onnxruntime, openai-whisper, conformer, x-transformers, hydra-core, inflect, pyworld, rich, gdown), BSD-3-Clause (omegaconf), ISC (librosa), MPL-2.0 và MIT (tqdm), Public Domain (wget), MIT (setuptools); chưa đọc lại từng file LICENSE gốc | được | chỉ cài khi chạy `vietvoice tts` trên máy người dùng (`pip install -e ".[tts]"`), người dùng tự tải từ PyPI |

## Chỉ dùng cho nghiên cứu (không được vào checkpoint sản phẩm)

Các nguồn dưới đây là CC-BY-NC hoặc tương tự: hynt F5, EraX, viVoice, ViMD, PhoAudiobook.

| Nguồn | Giấy phép | Dùng ở đâu (giai đoạn build, xem "train hết, giấy phép làm lại sau" 2026-11-23) |
| --- | --- | --- |
| PhoAudiobook (thivux/phoaudiobook, arXiv 2506.01322), bản đóng gói giahy2507/phoaudiobook_24kHz_part1 + part2 | "for research or educational purposes only", không được phân phối lại | vòng P (`infra/vast/prepare_pho.py`, `rounds/round_p.sh`): train **llm** (audio 16 kHz, không dùng cho flow). Mọi checkpoint từ `runs/llm-p-pho` trở đi đều nhiễm NC: chỉ nội bộ, phải train lại trước khi thương mại. Câu mẫu `voices/p/pho_h*` là người đọc sách nói chưa đồng ý: không đăng, không đặt tên công khai |
