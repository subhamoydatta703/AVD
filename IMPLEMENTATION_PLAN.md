# Revised implementation plan

Cleanup validation (2026-10-01): 19 offline tests and fresh Hindi/English smoke jobs passed. The five older jobs were deleted; source videos/audio were preserved with matching SHA-256 hashes under output/sources. Historical job paths below no longer exist. Current evidence: output/cleanup_verification.json. Full-video quality and listening review remain pending.

Current implementation update (2026-10-01): Gemini is the default translation provider. At the user's request it is one simple translate_text function in gemini_translation.py, returning translated text. The pipeline saves each translated segment separately. German/French routing and .env configuration are included. Legacy IndicTrans2, fastText detection and compatibility wrappers have been removed; Gemini is the only translation provider. The historical plan below records earlier findings, dependencies and test results, rather than the current installation; its proposal to improve/retest IndicTrans2 is superseded by the user's Gemini decision. Long-video submissions, voice cloning and listening review remain outstanding.

Updated 2026-10-01. This document replaces the earlier implementation sequence. It is a plan, not a claim that the remaining features work. No further model downloads or full-video runs are part of this planning step. No Git commands are permitted.

## Verified current status

| Area | Evidence and status |
| --- | --- |
| YouTube download | Both user-selected source videos are saved locally. |
| Audio extraction | FFmpeg produced readable mono 16 kHz PCM audio. The Hindi audio is nonempty and has measurable signal; this does not establish listening quality. |
| Hindi benchmark | Source duration 30:53. Whisper small failed the first five-minute chunk's script check. No final dub exists for this source. |
| Bengali benchmark | Source duration 1:03:36. Full run was interrupted during transcription to investigate quality. No final dub exists for this source. |
| Whisper small | Rejected for these movie samples. A diagnostic rerun produced 75 Hindi segments, 45 with average log probability below -1; median approximately -3.41. |
| Whisper turbo | Downloaded and tested on four 30-second real-video excerpts. More coherent Hindi output, but missing dialogue and incorrect Hindi/Bengali words remain. Script checks passed; accuracy has not passed. |
| IndicTrans2 | Official distilled Indic-En 200M model loads and translates five control sentences using Transformers 4.51.3. Real-video translation quality has not been established. |
| IndicConformer | Configuration access returned HTTP 403 with the local account. Weights have not been downloaded or tested. This is a separate gate from IndicTrans2. |
| TTS and mux | Edge TTS and FFmpeg completed a 12-second synthetic Hindi control. Copied video packet hashes matched. This is not a long-video or original-voice validation. |
| Runtime | Python 3.12.10, CPU-only PyTorch 2.14.0, Transformers 4.51.3. CUDA is unavailable in this environment. |
| Tests | 14 offline regression tests passed. They test program behavior, not speech or translation accuracy. |
| Captions | Hindi original automatic captions and a Bengali subtitle track are available and downloaded for comparison. Their correctness requires review. |

Evidence: `output/benchmarks/runs.json`, `output/benchmarks/asr-probes/turbo/report.json`, `output/jobs/25d91cbe91571e10/transcription/raw_00000_review.json`, `output/model_probe.json` and the individual job reports.

## Reasons for changing the approach

1. The synthetic control was too easy to justify a movie-length run. Real samples must drive model selection.
2. Whisper small returned severely unreliable speech text on the Hindi source. Passing model loading is different from passing transcription accuracy.
3. The current adapter uses Python transcription defaults without explicit beam search/best-of settings. Installed Whisper code can exhaust temperature fallbacks and still return low-confidence text. Better decoding is worth testing; it is not a guaranteed fix.
4. Script validation catches some failures but accepts wrong words in the correct script and Latin text. Turbo's results demonstrate that script and confidence checks alone cannot certify accuracy.
5. The Bengali video's opening includes a Hindi promotional passage, as indicated by its subtitle text. Forcing Bengali across the entire source is unsuitable; language overrides must support time ranges.
6. Current translation processes ASR fragments separately. Sentence context, names, quantities, cultural terms and long-sentence handling need real-data validation.
7. Current timing preserves every segment start and stops when speech needs more than 1.35x compression. It has no automatic repair loop. This is an implementation limitation; neither long run reached this stage, so it is not an observed benchmark failure yet.
8. A single Edge stock voice cannot reproduce several original speakers or their energy. The existing system removes all music and effects with the source audio.
9. CPU execution and repeated decoding fallbacks make long runs expensive. Changing the inference engine may improve speed, but cannot by itself repair recognition errors.

Background music, overlapping voices and recording characteristics are possible contributors to ASR errors. They require listening and controlled comparisons before being treated as established causes.

## Phase 1 — Establish a real evaluation set

- Retain the downloaded originals and all failed-run evidence; stop full-length processing.
- Select 8–12 short passages per language from beginning, middle and end, including fast dialogue, different speakers, silence/music, numbers/names and language changes.
- Review reference text against the actual audio. Captions are supporting evidence, not automatically ground truth. Deduplicate rolling automatic captions.
- Record transcript completeness, word errors against reviewed references, names/numbers and processing time. Keep uncertain passages explicitly flagged.
- Review at least one complete two-minute excerpt per language after individual sample testing.

Pass condition: the selected ASR must preserve the meaning and all critical names/numbers on the reviewed set, with an explicit error report. No missing lines or obvious hallucinations may be silently accepted. Native-language review remains necessary to certify linguistic quality.

## Phase 2 — Select and verify ASR

Use a controlled comparison, rather than committing to a replacement on documentation alone:

| Candidate | Role | Download/access status | Important checks |
| --- | --- | --- | --- |
| Cached Whisper turbo | Low-cost comparison with explicit beam search and conservative decoding | Already downloaded; checkpoint 1,617,941,637 bytes | Hindi/Bengali meaning, omissions, language changes and actual CPU time |
| AI4Bharat IndicConformer 600M multilingual | Preferred Indian-language candidate to evaluate | Repository metadata totals 2,556,502,676 bytes; local access currently 403 | Real Windows/CPU ONNX inference, required asset files, language-specific output, timestamp strategy and memory |
| Full Whisper large-v3 through faster-whisper | Practical multilingual fallback if IndicConformer is unavailable or loses the comparison | Not downloaded; verify exact selected artifact size before download | Windows CPU INT8 compatibility, word timestamps, accuracy and memory; larger model quality must be demonstrated |

Do not use English-only Distil-Whisper as a Hindi/Bengali ASR replacement. Do not assume faster-whisper with the same weights improves accuracy: it is an execution alternative.

- Test candidate dependencies in an isolated ASR environment before modifying the working translation environment.
- Verify real import, model load and inference, then freeze package versions and model revisions. Do not install the documentation's dependency list blindly.
- Use voice activity detection to identify speech; preserve absolute video timestamps and short utterances. VAD does not detect language or distinguish speakers.
- Support reviewed time-range language overrides, with detection used as evidence rather than an unquestioned decision on very short utterances.
- Add bounded decoding retries, diagnostics for omissions/repetition/low confidence, and an explicit review queue for unresolved speech. Never manufacture words to satisfy a script check.
- Verify timestamps on real dialogue. IndicConformer's example returns text; its example does not establish usable word timestamps. Evaluate VAD phrase boundaries or an appropriate alignment method before selecting it for dubbing.

Pass condition: a candidate wins the real evaluation, works in the actual environment, and produces trustworthy phrase timing. Record its measured memory and CPU cost before the long runs.

## Phase 3 — Improve IndicTrans2 translation

- Keep the verified IndicTrans2 200M adapter initially. ASR is the demonstrated bottleneck; replacing translation weights first would not fix it.
- Merge utterance fragments into sentences while retaining their constituent timestamps and language labels. Avoid joining across speaker turns where identifiable.
- Split oversized input at linguistic boundaries and translate all pieces; never silently truncate.
- Route each Indian-language passage with the correct FLORES tag and retain English passages directly.
- Review names, numbers, jokes and context-dependent lines. Preserve a traceable source-to-English mapping and corrections.
- Compare IndicTrans2 1B only if correct source text still yields unacceptable translations. Check access, exact size and real compatibility first.

Pass condition: both two-minute excerpts have reviewed, meaning-preserving English, including critical names and quantities. The five existing probe sentences alone do not satisfy this condition.

## Phase 4 — Make synthesis and timing reliable

- First validate a natural English stock voice as the core assignment baseline.
- Synthesize complete phrases, not arbitrary ASR fragments. Cache by text, voice and synthesis configuration.
- Measure actual speech duration before assembly and report all timing conflicts together.
- Repair long phrases through meaning-preserving shorter wording, suitable TTS speaking rate and limited pitch-preserving tempo adjustment. Do not truncate speech or raise compression indiscriminately.
- Permit small, bounded timing shifts inside reviewed utterance windows; preserve pauses and prevent cumulative drift. Escalate overlapping dialogue for review instead of silently discarding a speaker.
- Use bounded service retries and resumable work. Test modest concurrency only after service reliability is established.

Pass condition: both two-minute excerpts sound natural, retain all translated dialogue, preserve useful pauses and show no overlaps, truncation or accumulating timing drift. The tempo threshold alone is not an audible-quality test.

## Phase 5 — Original speaker identity and energy

The assignment's core checklist permits natural stock TTS, while the opening description asks for the original voice and energy. A stock-voice baseline must not be presented as achieving those richer requirements.

- After core accuracy and timing pass, evaluate diarization and a cross-language voice-cloning/voice-conversion candidate on clean references from these sources.
- Check the actual model's license, Windows/CPU support, English output, Hindi/Bengali reference compatibility and measured resource needs before selecting or downloading it.
- Test speaker consistency, expressive delivery and timing. Assign distinct reviewed voices if cloning cannot meet quality targets; clearly report that compromise.
- Evaluate dialogue/background separation if music and effects need to survive replacement. Listen for residual original speech and separation artifacts before using the background track.

Pass condition for the richer target: distinguishable, consistent speakers and reviewed voice/energy preservation. A model's cloning claim is not proof of achieving this on these videos.

## Phase 6 — Long-video benchmarks and delivery

- Run a 10-minute pilot for each language after the two-minute outputs pass. Test resume after a controlled interruption without repeating completed work.
- Then process the full Hindi 30:53 video and the supplied Bengali 1:03:36 video.
- Obtain a separate two-hour source for the assignment. Do not duplicate or pad the one-hour Bengali video to simulate this requirement.
- Copy the original video stream with FFmpeg. Verify packet hashes, duration, playable English audio and completion of every approved speech unit.
- Review beginning/middle/end, scene boundaries, fastest dialogue and every previously flagged passage. Technical validation is separate from linguistic and listening review.
- Record per-stage time, model setup, failed/resumed attempts, successful processing time, hardware and model revisions without mixing synthetic-control timings into benchmark claims.
- Prepare the originals, reviewed dubs, timing reports and a two-minute walkthrough demonstrating before/after playback and explaining verified decisions and limitations.

Pass condition: required-duration deliverables are complete and reviewed, with honest time records and the walkthrough. Email submission is a separate user-authorized action.

## Immediate next implementation order

1. Build the reviewed real-video evaluation set and language-range map.
2. Compare cached turbo with explicit decoding against IndicConformer if access is available, or full large-v3 as fallback; choose on measured accuracy and compatibility.
3. Produce and review two-minute Hindi and Bengali dubs, improving translation grouping and timing repair as required.
4. Complete ten-minute pilots, then the full benchmarks.
5. Add voice identity/energy improvements after the core passes; obtain the outstanding two-hour source and prepare final deliverables.

References for feasibility checks: [IndicConformer model card](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual), [OpenAI Whisper](https://github.com/openai/whisper), [faster-whisper documentation](https://github.com/SYSTRAN/faster-whisper), [IndicTrans2 200M](https://huggingface.co/ai4bharat/indictrans2-indic-en-dist-200M). These describe candidates; they do not replace tests in this project's environment.
