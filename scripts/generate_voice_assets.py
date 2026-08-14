#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["edge-tts>=7,<8"]
# ///

from __future__ import annotations

import argparse
import asyncio
import subprocess
import tempfile
from pathlib import Path

import edge_tts

PIPER_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PIPER_ROOT / "assets/sounds"

FIXED_CUES = {
    "ready": "조작 준비",
    "recording_start": "녹화 시작",
    "keyframe": "키프레임",
    "environment_reset": "환경 초기화",
    "rerecord": "재녹화",
    "acquisition_end": "취득 종료",
    "support_arms": "로봇팔 지지",
    "disconnected": "연결 해제",
    "upload_complete": "업로드 완료",
}


async def generate(
    *,
    voice: str,
    rate: str,
    episode_voice: str,
    episode_rate: str,
    max_episodes: int,
    max_keyframes: int,
    force: bool,
    force_episodes: bool,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cues = {
        name: (text, voice, rate, False)
        for name, text in FIXED_CUES.items()
    }
    cues.update(
        {
            f"recording_{episode}": (
                f"Recording episode {episode}",
                episode_voice,
                episode_rate,
                True,
            )
            for episode in range(1, max_episodes + 1)
        }
    )
    countdown_words = {3: "Three", 2: "Two", 1: "One"}
    cues.update(
        {
            f"countdown_{number}": (
                word,
                episode_voice,
                "+5%",
                False,
            )
            for number, word in countdown_words.items()
        }
    )
    cues.update(
        {
            f"keyframe_{keyframe}": (
                f"키프레임 {keyframe}",
                voice,
                rate,
                False,
            )
            for keyframe in range(1, max_keyframes + 1)
        }
    )

    with tempfile.TemporaryDirectory(prefix="piper-voice-") as temporary:
        temp_dir = Path(temporary)
        for name, (text, cue_voice, cue_rate, is_episode) in cues.items():
            mp3_path = temp_dir / f"{name}.mp3"
            wav_path = OUTPUT_DIR / f"{name}.wav"
            if wav_path.is_file() and not force and not (force_episodes and is_episode):
                print(f"{name:<20} {text} (cached)")
                continue
            await edge_tts.Communicate(text, cue_voice, rate=cue_rate).save(mp3_path)
            command = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(mp3_path),
            ]
            if name.startswith("countdown_"):
                command.extend(
                    [
                        "-af",
                        (
                            "silenceremove=start_periods=1:start_duration=0.02:"
                            "start_threshold=-45dB:stop_periods=1:stop_duration=0.10:"
                            "stop_threshold=-45dB"
                        ),
                    ]
                )
            command.extend(
                [
                    "-ar",
                    "48000",
                    "-ac",
                    "1",
                    "-c:a",
                    "pcm_s16le",
                    str(wav_path),
                ]
            )
            subprocess.run(command, check=True)
            print(f"{name:<20} {text}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate cached Piper status WAV files.")
    parser.add_argument("--voice", default="ko-KR-SunHiNeural")
    parser.add_argument("--rate", default="-5%")
    parser.add_argument("--episode-voice", default="en-US-JennyNeural")
    parser.add_argument("--episode-rate", default="-10%")
    parser.add_argument("--max-episodes", type=int, default=50)
    parser.add_argument("--max-keyframes", type=int, default=10)
    parser.add_argument("--force", action="store_true", help="regenerate existing WAV files")
    parser.add_argument(
        "--force-episodes",
        action="store_true",
        help="regenerate only the numbered English episode WAV files",
    )
    args = parser.parse_args()
    if args.max_episodes < 1:
        parser.error("--max-episodes must be positive")
    if args.max_keyframes < 1:
        parser.error("--max-keyframes must be positive")
    asyncio.run(
        generate(
            voice=args.voice,
            rate=args.rate,
            episode_voice=args.episode_voice,
            episode_rate=args.episode_rate,
            max_episodes=args.max_episodes,
            max_keyframes=args.max_keyframes,
            force=args.force,
            force_episodes=args.force_episodes,
        )
    )


if __name__ == "__main__":
    main()
