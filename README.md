# Structured reads on DiffusionGemma

A discrete diffusion model denoises a whole canvas per forward pass. If the
canvas is seeded with the answer's fixed text and only the answer slots are
left as noise, one denoise step gives a distribution over each slot. These
`extra_args` fields (`vllm_xargs` on the OpenAI server) expose that:

| field | type | meaning |
| --- | --- | --- |
| `diffusion_seed_canvas` | `list[int]`, exactly `canvas_length` ids | replaces the random initial canvas after prefill |
| `diffusion_pinned` | `list[int]` of canvas positions | held at their seed value on every denoise step, so a read past one step keeps its template |
| `diffusion_max_steps` | `int` | denoise steps before the canvas is emitted |
| `diffusion_read_only` | `bool` | emit the argmax canvas as soon as the cap is reached, end the request there, and return temperature-1 logprobs at every position |

`structured_server.py` turns a question schema into those fields, or, with
`--engine ar`, into next-token reads on any vLLM model that returns
logprobs (below). It serves
`/v1/chat/completions`: the system message is the schema, the user message
is the state JSON, and the reply content is one distribution per question
with a standard error over a few noise draws.

```bash
vllm serve google/diffusiongemma-26B-A4B-it \
    --diffusion-config '{"canvas_length": 64}' --max-logprobs 32 --enable-prefix-caching
python examples/features/structured_diffusion/structured_server.py \
    --upstream http://127.0.0.1:8000 --tokenizer google/diffusiongemma-26B-A4B-it --canvas 64
curl -s localhost:8011/v1/chat/completions -H 'content-type: application/json' -d '{
  "messages": [
    {"role": "system", "content": "{\"questions\": [{\"id\": \"urgent\", \"type\": \"noul\", \"instructions\": \"Does the customer need a reply within the hour?\"}]}"},
    {"role": "user", "content": "{\"ticket\": \"Everything is down and we have a demo at noon.\"}"}
  ]}'
```

Per-request canvas widths smaller than the served canvas require async
scheduling. The diffusion async scheduler is selected automatically; no
`--scheduler-cls` argument is needed. Synchronous execution supports full-width
canvases only.

Question types: `noul` (yes/no), `choice` with `options`, `score` with
ordered `levels`, and `span` / `spans` (below). Each label must be a single
token in the answer template, which the server checks with the tokenizer
when a request uses the schema.

`POST /v1/systemone` implements the Jev decision API. The body holds
`state`, `questions` (a map of id to `type`, `instructions` and `criteria`)
and `model`. Answers come back in that API's shapes: a `noul` probability,
a `choice` with `probabilities` and `confidence`, or a `score` with a
0-indexed `legend`. The schema options above go in the same body as
extensions.

A question may declare `depends_on` (read in a later stage with those
answers in its prompt), `ask_if` (asked only when a named question's answer
is among the listed ones, otherwise null) and `alone` (a read of its own).

Images attach as `multipart/form-data`, with the JSON in a part named
`request` and each image as a file part, or as an `images` array of data
URLs.

```bash
curl -s localhost:8011/v1/systemone -H 'content-type: application/json' -d '{
  "model": "jev-latest",
  "state": {"ticket": "Everything is down and we have a demo at noon."},
  "questions": {"urgent": {"type": "noul", "instructions": "Does the customer need a reply within the hour?"}}}'
```

## Any vLLM model with logprobs

`--engine ar` runs the same schemas on an ordinary autoregressive model
served by vLLM, with no diffusion machinery. A label is one next-token read
restricted to the question's label ids, taken in question order with the
earlier answers prefilled, so a request with three questions is three reads
of a few tokens each behind a cached prefix. A span is one restricted
first-token read to fix the start, then every candidate end within twelve
words scored teacher-forced through `prompt_logprobs`, the best total
against a `none` candidate. A `spans` question asks for the first, second,
third value until none. Reads are deterministic, so `samples` collapses to
one; `think` and images are not available on this engine.

Without `--tokenizer` the server tokenizes through the upstream's own
`/tokenize` and `/detokenize`, so nothing has to be installed locally:

```bash
python structured_server.py --engine ar --upstream http://host:8000 --model glm53 --port 8011
```

Measured against GLM 5.3 (`tests/label_battery.py`, `tests/span_battery.py`),
with DiffusionGemma NVFP4 on the diffusion engine beside it:

| | GLM 5.3, `--engine ar` | DiffusionGemma, diffusion |
| --- | --- | --- |
| 22 label questions (noul, choice, score) | 21 | 20 |
| 49 span fields, 11 texts | 48 | 49 |
| 5 list questions | 5 | 5 |
| a 1096-character text in 4 windows, 4 fields | 3, in 89 s | 4, in 4 s |
| reads per label question | 1, about 0.27 s | one joint read for the schema, about 0.1 s |
| reads per span field | 1 plus one per candidate end, about 1 s | 1 to 3, about 0.35 s |

The AR span's one miss returned an email for an absent phone number at
confidence 0.10, so the usual gate catches it. The long text is where the AR
engine pays: every window scores its own candidate ends, so cost grows with
windows times candidates.

The diffusion engine's edge is cost: every label of a schema comes from one
canvas read, and a span from one read of the blank. The AR engine's edge is
that it needs nothing beyond stock vLLM.

## Span answers

`span` answers with a piece of the state's text, as character offsets, and
`spans` with every such piece. The model never writes the value: the read
seeds the reply canvas with a label and a blank, pins everything but the
blank, runs four denoise steps, and returns the logprobs of the text's own
token ids at every position of the blank. The decode walks the text with
those tokens' strings, so the answer is a substring by construction and
cannot be a value the text does not contain. `start` and `end` index the
state when it is a string, else its `"text"` field.

```bash
curl -s localhost:8011/v1/systemone -H 'content-type: application/json' -d '{
  "model": "jev-latest",
  "state": "Invoice #A-1042 from Northwind Traders. Total due: $1,234.56 by 2024-03-15.",
  "questions": {
    "invoice_id": {"type": "span", "instructions": "the invoice id"},
    "amount": {"type": "span", "instructions": "the total amount due"},
    "dates": {"type": "spans", "instructions": "date"}}}'
```

```json
{"invoice_id": {"type": "span", "found": true, "text": "A-1042", "start": 9, "end": 15,
                "confidence": 0.87, "coverage": 0.99, "reads": 1},
 "amount":     {"type": "span", "found": true, "text": "$1,234.56", "start": 51, "end": 60, ...},
 "dates":      {"type": "spans", "found": true, "items": [{"text": "2024-03-15", "start": 64, "end": 74, ...}]}}
```

`confidence` is the weakest normalised probability along the span and its
end; a boundary the model was unsure of reads 0.1 to 0.3 against 0.5 and up
on clean values. A read under 0.3 is repeated with the next seed, up to
three times, and the most confident answer kept. `coverage` is the share of
the model's probability mass that the text's tokens held along the span.
Criteria: `max_tokens` (the blank, default 24 for `span` and 64 for
`spans`) and `max_items`. Span questions run beside the label reads and
cannot take part in `depends_on` or `ask_if`. Values that must be
normalised (an ISO date from "Oct 1st") are not spans; extract the span and
convert it in code.

A text with more distinct tokens than one read can score (128) is split at
sentences into windows with one sentence of overlap. The prompt always
carries the whole text; only the read is restricted to each window, so the
model is never answering from a fragment. Windows are read in parallel, the
answer with the best confidence and coverage wins, and when windows
disagree one further one-step choice over the candidates settles it. A
`spans` question reads each window twice with different seeds and merges
the lines by offset, since a line one read omits another usually writes.

`tests/span_battery.py` runs 49 span fields, five lists, a mixed schema, the
chat form and a four-window text against a live server.

`"think": N` in the schema lets the model write up to N tokens in its
thought channel before the read. The thought is an ordinary generation with
the chat template's thinking marker on, and the read then runs with the
thought in its prompt, so the answer slots condition on it. The noise draws
of a decision share one thought. `diagnostics.thought` returns the text, its
length in tokens, whether the model closed the channel itself and the
generation time.

## Provenance

This server is the example that ships with vLLM PR
[#57250](https://github.com/vllm-project/vllm/pull/57250), which adds
structured reads to DiffusionGemma (seeded canvas, pinned positions, a step
cap, read-only requests and exact logprobs for chosen token ids). This
repository is its home; the copy in the PR is a snapshot. It runs in front of
a vLLM built from that branch.
