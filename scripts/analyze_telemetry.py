import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KORGAN telemetry JSONL analiz araci")
    parser.add_argument("--log", type=Path, help="Analiz edilecek telemetry JSONL dosyasi")
    parser.add_argument("--logs-dir", type=Path, default=Path("logs"), help="Log klasoru")
    parser.add_argument("--top", type=int, default=5, help="En sik hedefleri gosterme limiti")
    return parser.parse_args()


def latest_log(logs_dir: Path) -> Path:
    candidates = sorted(logs_dir.glob("telemetry_*.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"Telemetry log bulunamadi: {logs_dir}")
    return candidates[-1]


def avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * ratio))))
    return ordered[index]


def format_stats(values: list[float], suffix: str = "") -> str:
    if not values:
        return "-"
    return (
        f"avg {avg(values):.2f}{suffix} | "
        f"p95 {percentile(values, 0.95):.2f}{suffix} | "
        f"max {max(values):.2f}{suffix}"
    )


def load_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"JSON parse hatasi satir {line_number}: {exc}") from exc
            events.append(payload)
    return events


def main() -> int:
    args = parse_args()
    log_path = args.log if args.log else latest_log(args.logs_dir)
    events = load_events(log_path)
    if not events:
        raise RuntimeError(f"Bos telemetry log: {log_path}")

    event_counts: Counter[str] = Counter()
    tracking_counts: Counter[str] = Counter()
    primary_labels: Counter[str] = Counter()
    primary_profiles: Counter[str] = Counter()
    primary_classes: Counter[int] = Counter()
    unresolved_warnings: Counter[str] = Counter()
    timings: dict[str, list[float]] = defaultdict(list)
    fps_values: list[float] = []
    distances_m: list[float] = []
    lock_ready_distances_m: list[float] = []
    visible_primary_frames = 0
    lock_ready_frames = 0
    engagement_allowed_frames = 0
    worker_restart_max = 0

    first_ts = events[0].get("ts", "-")
    last_ts = events[-1].get("ts", "-")

    for event in events:
        event_name = str(event.get("event", "unknown"))
        event_counts[event_name] += 1

        if event_name == "target_profile_warning":
            unresolved_warnings[str(event.get("message", "-"))] += 1

        if event_name == "worker_restarted":
            worker_restart_max = max(worker_restart_max, int(event.get("restart_count", 0)))

        if event_name != "frame":
            continue

        fps = event.get("fps")
        if isinstance(fps, (int, float)):
            fps_values.append(float(fps))

        tracking_status = str(event.get("tracking_status", "BILINMIYOR"))
        tracking_counts[tracking_status] += 1

        worker_restart_max = max(worker_restart_max, int(event.get("worker_restart_count", 0)))

        for key, value in event.get("timings_ms", {}).items():
            if isinstance(value, (int, float)):
                timings[str(key)].append(float(value))

        primary = event.get("primary_target")
        if not isinstance(primary, dict):
            continue

        visible_primary_frames += 1
        label_name = str(primary.get("label", "-"))
        if label_name and label_name != "-":
            primary_labels[label_name] += 1

        profile_name = str(primary.get("profile", "-"))
        if profile_name and profile_name != "-":
            primary_profiles[profile_name] += 1

        class_id = primary.get("class_id")
        if isinstance(class_id, int):
            primary_classes[class_id] += 1

        distance_m = primary.get("distance_m")
        if isinstance(distance_m, (int, float)):
            distances_m.append(float(distance_m))

        in_lock_window = bool(primary.get("in_lock_window", False))
        engagement_allowed = bool(primary.get("engagement_allowed", in_lock_window))

        if in_lock_window:
            lock_ready_frames += 1
            if isinstance(distance_m, (int, float)):
                lock_ready_distances_m.append(float(distance_m))

        if engagement_allowed:
            engagement_allowed_frames += 1

    total_frames = event_counts.get("frame", 0)

    print(f"Log: {log_path.resolve()}")
    print(f"Zaman araligi: {first_ts} -> {last_ts}")
    print("")

    print("Olaylar")
    for name, count in sorted(event_counts.items()):
        print(f"- {name}: {count}")
    print("")

    print("Performans")
    print(f"- FPS: {format_stats(fps_values)}")
    for key in ["round_trip", "decode", "preprocess", "inference", "postprocess", "worker_total", "tracker", "render", "frame_total"]:
        if key in timings:
            print(f"- {key}: {format_stats(timings[key], ' ms')}")
    print("")

    print("Takip")
    print(f"- Frame sayisi: {total_frames}")
    print(f"- Gorunen birincil hedef: {visible_primary_frames}/{total_frames}")
    print(f"- Kilit hazir frame: {lock_ready_frames}/{total_frames}")
    print(f"- Angajman uygun frame: {engagement_allowed_frames}/{total_frames}")
    print(f"- Worker restart max: {worker_restart_max}")
    if tracking_counts:
        print("- Durum dagilimi:")
        for status, count in tracking_counts.most_common():
            print(f"  {status}: {count}")
    print("")

    print("Hedefler")
    if primary_profiles:
        print("- En sik profiller:")
        for name, count in primary_profiles.most_common(args.top):
            print(f"  {name}: {count}")
    if primary_labels:
        print("- En sik etiketler:")
        for name, count in primary_labels.most_common(args.top):
            print(f"  {name}: {count}")
    if distances_m:
        print(f"- Hedef mesafesi: {format_stats(distances_m, ' m')}")
    if lock_ready_distances_m:
        print(f"- Kilit hazir mesafeleri: {format_stats(lock_ready_distances_m, ' m')}")
    print("")

    if unresolved_warnings:
        print("Uyarilar")
        for message, count in unresolved_warnings.items():
            print(f"- {message}: {count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
