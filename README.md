# SwitchBench: Mid-Execution Spoken Command Switching

**Can a robot that is already acting change what it is doing when you tell it to?**

Speech-conditioned vision-language-action models such as ELLSA (End-to-end Listen, Look,
Speak and Act, ICLR 2026) listen while they act. ELLSA was evaluated on starting a task from
speech and on *stopping* when interrupted ("Hold on"), but not on the next natural step:
**switching to a different task mid-execution** ("Actually, put the bowl on the plate
instead"), while **ignoring** speech that is not a command ("Okay", "Uh-huh, go on").
SwitchBench measures exactly that.

## Task

Each episode is a LIBERO-Goal manipulation episode driven only by the user's voice.

| type | what the user says | success |
|---|---|---|
| **switch** | instruction for task A, then after a gap a command to do task B instead | task B's goal is reached after the command starts |
| **keep** | instruction for task A, then an irrelevant utterance (backchannel) | task A's goal is reached |
| **plain** | instruction for task A only | task A's goal is reached |

The policy receives **camera images, proprioception and the raw 16 kHz audio heard so far**.
It never receives text (no task name, no transcript) and never hears future audio.

### Timeline (1 env step = 0.1 s)

```
t=0          instr_end        utt_start     utt_end                    max_steps
|-- instruction --|--- gap ------|-- utterance --|------ 300-step budget ------|
```

`gap` is 2, 4 or 6 s. Plain episodes get `instr_end + 300` steps. The environment first runs
10 no-op steps so objects settle (as in the official LIBERO protocol); these are not counted.

### Why LIBERO-Goal

All 10 LIBERO-Goal tasks share one kitchen scene and the same four objects, so any task B
can be requested from any task A's starting state. We verified for all 90 ordered pairs that
object poses are identical across task environments for a given initial state and that no
task's goal is already satisfied at the start (`checks/`). Success of any task is therefore
judged by LIBERO's own BDDL goal predicates, with no hand-written success code.

## Metrics

**Score = (SSR + KSR) / 2**

* **SSR**, switch success rate: switch episodes where B was completed.
* **KSR**, keep success rate: keep episodes where A was completed despite the utterance.

A policy that reacts to every utterance fails *keep*; one that ignores speech mid-task fails
*switch*. Diagnostics reported alongside (not in the score): **PSR** (plain success, i.e.
basic speech-conditioned manipulation), **stale completion** (finished the old task A after
being told to switch), **A done before cue**, **median steps from end of cue to success**,
and SSR per gap. All rates carry Wilson 95% confidence intervals.

PSR is always shown on the leaderboard. It breaks ties on Score, and a run with PSR below 60%
(the lower 95% bound of zero-shot ELLSA on dev, 9/10) is flagged ⚠: it may have traded basic
manipulation for switching. PSR is kept out of the Score because plain episodes have no second
utterance, so they add the same points to every method built on the same base model.

## Data (v1.0)

| split | episodes | initial states | speech | use |
|---|---|---|---|---|
| **train** | 960 switch + 600 keep + 200 plain | 40 new states per task (`assets/train_init_states.npz`) | ELLSA *train* recordings, VoiceAssistant-400K questions, gTTS | public, for training |
| dev | 30 switch + 10 keep + 10 plain | LIBERO 0–24 | ELLSA *test* recordings, question set D | public, for tuning |
| test | 30 switch + 10 keep + 10 plain | LIBERO 25–49 | ELLSA *test* recordings, question set T | hidden, for ranking |

All splits share one episode format (`manifest.json` + `audio/`). Dev and test place every task
once as A and once as B (a random derangement) at gaps of 2, 4 and 6 s; the switch cue is task B's
own instruction recording and the keep cue a spoken general question.

**SwitchBench-train** (`scripts/build_train.py`, no GPU or simulator needed):

* commands: the per-demo instruction recordings in the ELLSA *train* data (many voices, ELLSA's
  own training distribution), plus gTTS paraphrases ("Actually, ... instead");
* non-commands: spoken questions from VoiceAssistant-400K, plus gTTS backchannels and questions
  about the scene's objects (hard negatives);
* new LIBERO-Goal initial states from fresh seeds, none equal to a dev/test state and none with a
  goal already met; all 80 ordered task pairs except the 10 test pairs;
* every clip is checked against dev/test (byte hash and loudness-envelope match) and every question
  text against the dev/test questions; overlaps are dropped and listed in `leak_report.json`.

Labels follow from the episode type (switch cue = command, keep cue = not a command), so the same
split trains a speech gate (audio + labels) or a policy (run the episodes in the simulator).

## Rules

* Train only on SwitchBench-train and public pretrained models (ELLSA, Whisper, ...).
* Use dev for tuning; run test once for the final score.
* At test time a policy gets images, robot state and the audio heard so far: no task ids, texts or
  future audio.

## Baseline: ELLSA (zero-shot)

`baselines/ellsa_policy.py` wraps the public ELLSA checkpoint without any training. Its model
call mirrors the official LIBERO evaluation (same flags, 1 s blocks of 10 actions); the only
change is that the future audio is replaced by silence so the policy cannot look ahead.

## Leaderboard (dev, v0 official recordings)

| # | model | training | Score | SSR (switch) | KSR (keep) | PSR (plain) | stale completion |
|---|---|---|---|---|---|---|---|
| B0 | ELLSA (public checkpoint) | none (zero-shot) | **40.0** | 0.0% (0/30) | 80.0% (8/10) | 90.0% (9/10) | 90.0% (27/30) |
| - | NoopPolicy (never moves) | none | 0.0 | 0% | 0% | 0% | 0% |
| B1 | Restart ELLSA on any utterance | none | not run | | | | |
| P1 | Rule-based intent (Whisper transcript + verbs) + restart | none | not run | | | | |
| P2 | Learned speech intent gate + restart (proposed) | SwitchBench-train (gate) | 40.0 | 0.0% (0/30) | 80.0% (8/10) | 90.0% (9/10) | 56.7% (17/30) |

ELLSA completes ordinary spoken tasks (PSR 90%, in line with its reported 86.4% on LIBERO-Goal)
but never switched to a new task mid-execution; in 27 of 30 switch episodes it finished the
original task instead. The command always arrived before task A was done, at every gap.
Cost: ~1–4 min per episode on one A100 80GB, ~2.5 h for the dev split.

**Data version note.** v0 uses the recordings shipped with the ELLSA test data: the switch cue is
task B's own instruction recording, the keep cue is a spoken general question. With Google TTS
voices ELLSA rejected even valid instructions ("Action cancelled"), so synthetic voices
(`--engine cosyvoice|gtts`) are kept as a robustness condition for a later version.

## Running

```bash
# 1) data (GPU for TTS; --fake for a no-GPU pipeline test)
python scripts/build_dataset.py --split dev --out data/switchbench \
    --prompt_dir $ELLSA_DATA_PATH --cosy $COSY_CKPT_PATH --ellsa_base $ELLSA_BASE_PATH
# 2) evaluate (resumable)
python scripts/run_eval.py --data data/switchbench/dev --policy ellsa --out results/ellsa_dev
# 3) score any results file
python scripts/score.py results/ellsa_dev/episodes.jsonl
```

LIBERO must be importable (`PYTHONPATH` contains the LIBERO repo) and rendering configured
(`MUJOCO_GL=egl` on GPU, `osmesa` on CPU). The ELLSA baseline additionally needs the ELLSA
environment from the setup notebook.

## Submitting a model

Implement `switchbench.policy.Policy`:

```python
class MyPolicy(Policy):
    def reset(self, info): ...          # info: max_steps, sample_rate, step_sec, episode_id
    def act(self, obs, audio, t):       # obs: LIBERO observation dict, audio: float32 16 kHz so far
        return {"actions": np.ndarray of shape (k, 7), "text": optional str}
```

and run `scripts/run_eval.py --policy my_module:MyPolicy` (see Rules above). Report the dev score
and submit `episodes.jsonl` from the test split; `scripts/leaderboard.py` ranks result folders.

## Sanity checks shipped with the code

* `NoopPolicy` (never moves) and `RandomPolicy` validate the harness end-to-end without a GPU.
* `checks/libero_goal_pair_check.txt`: cross-task scene equivalence for all 90 pairs.
