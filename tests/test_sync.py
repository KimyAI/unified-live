from unified_live.sync import SyncEngine, TimestampBuffer


def test_sync_delays_faster_pipeline_from_uncompensated_samples():
    sync = SyncEngine(alpha=1, max_delay_ms=500)
    result = sync.update(30, 90)
    assert result["valid"]
    assert result["video_delay_ms"] == 60
    assert result["audio_delay_ms"] == 0
    result = sync.update(None, None, video_offset_ms=-20, audio_offset_ms=20)
    assert result["video_delay_ms"] == 20
    assert result["audio_delay_ms"] == 0
    assert result["video_baseline_ms"] == 30
    assert result["audio_baseline_ms"] == 90
    sync.reset()
    assert sync.update(30, None)["valid"] is False


def test_buffer_bounded_drops_and_schedules_from_capture_time():
    buf = TimestampBuffer[str](max_items=2, max_age_ms=1000)
    buf.push(10.0, "old")
    buf.push(10.1, "a")
    buf.push(10.2, "b")
    assert buf.dropped == 1
    assert buf.pop_ready(10.14, 50) is None
    assert buf.pop_ready(10.15, 50) == "a"
    assert buf.pop_ready(10.26, 50) == "b"
    buf.push(11, "stale")
    assert buf.pop_ready(12.1) is None
    assert buf.dropped == 2
