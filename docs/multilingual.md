# Multilingual data and decoding

NUTQ uses two target representations:

- the Whisper multilingual tokenizer for the AR decoder;
- UTF-8 bytes for CTC and NUTQ-X TDT.

The tokenizer can represent any UTF-8 text, but recognition quality exists only for
languages and domains covered by the backbone and training data.

## Single-language training

Set one default language in the recipe:

```yaml
model:
  language: uzbek
  task: transcribe

data:
  language_column: null
  task_column: null
  default_language: uzbek
  default_task: transcribe
```

The manifest needs only audio and text:

```json
{"audio": "/data/uz/0001.wav", "text": "Bugun havo yaxshi."}
```

## Multilingual training

Use per-example language/task columns:

```yaml
data:
  audio_column: audio
  text_column: text
  language_column: language
  task_column: task
  default_language: null
  default_task: transcribe
```

```json
{"id":"uz-0001","audio":"/data/uz/0001.wav","text":"Bugun havo yaxshi.","language":"uzbek","task":"transcribe","speaker_id":"uz-spk-12","duration":3.42}
{"id":"ru-0001","audio":"/data/ru/0001.wav","text":"Сегодня хорошая погода.","language":"russian","task":"transcribe","speaker_id":"ru-spk-08","duration":2.91}
{"id":"en-0001","audio":"/data/en/0001.wav","text":"The weather is good today.","language":"english","task":"transcribe","speaker_id":"en-spk-03","duration":2.55}
```

`id`, `speaker_id`, `duration`, `source`, and `domain` are recommended manifest metadata for
validation, filtering, bucketing, and split audits. The current model input consumes the
configured audio/text/language/task fields.

Language accepts the names or codes supported by the selected Whisper tokenizer, for example
`uzbek`/`uz`, `russian`/`ru`, and `english`/`en`. An invalid value fails during preprocessing;
it is not silently mapped.

For multilingual sampling, balance by recorded hours or examples in the dataset pipeline.
NUTQ does not route named phrases or scripts through hardcoded language branches.

## Code-switching

A code-switched transcript may contain any UTF-8 script. Set `language` to the dominant or
dataset-defined Whisper language. Keep separate code-switch evaluation slices; tokenizer
coverage alone does not guarantee code-switch accuracy.

## Transcription and translation

CTC/TDT are monotonic speech-to-text objectives and are trained only for records whose
`task` is `transcribe`. Translation records train the AR decoder only. This masking happens
per example, so transcription and translation records may coexist in one dataset.

## Inference language selection

Force a known language for lower detection overhead and consistent evaluation:

```bash
nutq transcribe audio.wav --model outputs/nutq-m --strategy ar \
  --language uzbek --task transcribe
```

Omit `--language` to let the Whisper AR path detect the language:

```bash
nutq transcribe audio.wav --model outputs/nutq-m --strategy ar
```

CTC and TDT emit UTF-8 bytes and do not run Whisper's language-detection step. Their language
coverage comes from training data. In NUTQ-X `auto` mode, the AR fallback uses the supplied
language or detects it when no language is supplied.
