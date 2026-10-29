"""Build a SwitchBench split: manifest + synthesized audio.

    python scripts/build_dataset.py --split dev --out data/switchbench \
        --prompt_dir $ELLSA_DATA_PATH --cosy $COSY_CKPT_PATH --ellsa_base $ELLSA_BASE_PATH

Output: <out>/<split>/manifest.json and <out>/<split>/audio/*.wav
--fake writes placeholder tones instead of speech (pipeline test, no GPU needed).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from switchbench import manifest, tts  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=list(manifest.SPLITS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt_dir", help="folder with speaker prompt wavs (searched recursively)")
    ap.add_argument("--cosy", help="CosyVoice2-0.5B checkpoint folder")
    ap.add_argument("--ellsa_base", help="ELLSA repo (for its vendored CosyVoice code)")
    ap.add_argument("--asr", default="base.en", help="Whisper model for the intelligibility check ('' = off)")
    ap.add_argument("--engine", default="cosyvoice", choices=["cosyvoice", "gtts", "official"],
                    help="cosyvoice (zero-shot voices), gtts (accents as speakers), or official "
                         "(ELLSA test-data recordings: in-domain for ELLSA; needs --prompt_dir = ELLSA test data)")
    ap.add_argument("--fake", action="store_true")
    args = ap.parse_args()

    out_dir = os.path.join(args.out, args.split)
    os.makedirs(out_dir, exist_ok=True)
    episodes = manifest.build_manifest(args.split)
    if args.engine == "official":
        from switchbench import official_audio
        clips = official_audio.build(episodes, out_dir, args.prompt_dir, args.split)
        json.dump({"engine": "official", "asr_check": None}, open(os.path.join(out_dir, "tts_info.json"), "w"))
        manifest.save(episodes, os.path.join(out_dir, "manifest.json"))
        counts = {t: sum(e["type"] == t for e in episodes) for t in ("switch", "keep", "plain")}
        print(f"wrote {out_dir}/manifest.json: {len(episodes)} episodes {counts}, {len(clips)} audio clips (official)")
        for e in episodes[:5]:
            print(f"  {e['id']}: '{e['instruction_text']}' -> '{e['utterance_text']}'")
        return
    n_spk = max(e["speaker"] for e in episodes) + 1
    if args.fake or args.engine == "gtts":
        prompts = [str(i) for i in range(n_spk)]
    else:
        prompts = tts.find_prompts(args.prompt_dir, n_spk)
        json.dump(prompts, open(os.path.join(out_dir, "speaker_prompts.json"), "w"), indent=1)
    synth = tts.Synth(args.cosy, args.ellsa_base, fake=args.fake, asr_model=args.asr or None, engine=args.engine)
    json.dump({"engine": "fake" if args.fake else args.engine, "asr_check": args.asr or None},
              open(os.path.join(out_dir, "tts_info.json"), "w"))
    clips = tts.synthesize_episodes(episodes, out_dir, prompts, synth)
    manifest.save(episodes, os.path.join(out_dir, "manifest.json"))

    counts = {t: sum(e["type"] == t for e in episodes) for t in ("switch", "keep", "plain")}
    bad = [c for c in clips.values() if c["asr_wer"] is not None and c["asr_wer"] > 0.25]
    print(f"\nwrote {out_dir}/manifest.json: {len(episodes)} episodes {counts}, {len(clips)} audio clips"
          f"{'' if not bad else f', {len(bad)} clips above WER 0.25 (listed in audio/clips.json)'}")


if __name__ == "__main__":
    main()
