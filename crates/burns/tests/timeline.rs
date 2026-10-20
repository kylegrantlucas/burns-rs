use burns::timeline::{FrameRate, Timeline};

const NTSC: FrameRate = FrameRate::new(30000, 1001);

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-12
}

#[test]
fn ntsc_duration_rounds_to_whole_frames_and_reports_actual_duration() {
    let timeline = Timeline::from_duration(NTSC, 8.0).unwrap();
    // 8 s * 29.97 fps = 239.76 frames.
    assert_eq!(timeline.frames, 240);
    assert!(close(timeline.duration_seconds(), 240.0 * 1001.0 / 30000.0));

    let frame = timeline.frame_time(1).unwrap();
    assert!(close(frame.seconds, 1001.0 / 30000.0));
    let last = timeline.frame_time(239).unwrap();
    assert!(close(last.seconds, 239.0 * 1001.0 / 30000.0));
}

#[test]
fn progress_spans_endpoints_and_phase_loops() {
    let timeline = Timeline::new(FrameRate::new(30, 1), 5).unwrap();
    let first = timeline.frame_time(0).unwrap();
    let last = timeline.frame_time(4).unwrap();
    assert_eq!((first.progress, first.phase), (0.0, 0.0));
    assert_eq!(last.progress, 1.0);
    assert_eq!(last.phase, 0.8);
    assert_eq!(timeline.frame_time(2).unwrap().progress, 0.5);
}

#[test]
fn single_frame_timeline_has_zero_progress_and_phase() {
    // 0.001 s at 30 fps rounds to zero frames and clamps to one.
    let timeline = Timeline::from_duration(FrameRate::new(30, 1), 0.001).unwrap();
    assert_eq!(timeline.frames, 1);
    let frame = timeline.frame_time(0).unwrap();
    assert_eq!(
        (frame.seconds, frame.progress, frame.phase),
        (0.0, 0.0, 0.0)
    );
}

#[test]
fn frame_index_at_or_past_count_is_rejected() {
    let timeline = Timeline::new(FrameRate::new(24, 1), 10).unwrap();
    assert!(timeline.frame_time(9).is_ok());
    let error = timeline.frame_time(10).unwrap_err();
    assert_eq!(error.field(), "frame");
    assert!(timeline.frame_time(u32::MAX).is_err());
}

#[test]
fn invalid_durations_are_rejected() {
    let fps = FrameRate::new(30, 1);
    for seconds in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let error = Timeline::from_duration(fps, seconds).unwrap_err();
        assert_eq!(error.field(), "duration", "{seconds}");
    }
    // Frame count overflow: u32::MAX fps for 2 s, and an enormous duration.
    let overflow = Timeline::from_duration(FrameRate::new(u32::MAX, 1), 2.0).unwrap_err();
    assert_eq!(overflow.field(), "duration");
    assert!(Timeline::from_duration(fps, f64::MAX).is_err());
    // The largest representable count is still accepted.
    let max = Timeline::from_duration(FrameRate::new(u32::MAX, 1), 1.0).unwrap();
    assert_eq!(max.frames, u32::MAX);
}

#[test]
fn zero_frame_rate_terms_and_zero_frames_are_rejected() {
    assert_eq!(
        Timeline::from_duration(FrameRate::new(0, 1), 1.0)
            .unwrap_err()
            .field(),
        "fps.num"
    );
    assert_eq!(
        Timeline::from_duration(FrameRate::new(30, 0), 1.0)
            .unwrap_err()
            .field(),
        "fps.den"
    );
    assert_eq!(
        Timeline::new(FrameRate::new(30, 1), 0).unwrap_err().field(),
        "frames"
    );
    let invalid = Timeline {
        fps: FrameRate::new(30, 1),
        frames: 0,
    };
    assert!(invalid.frame_time(0).is_err());
}

#[test]
fn retiming_preserves_duration() {
    let timeline = Timeline::new(FrameRate::new(30, 1), 240).unwrap();
    assert_eq!(timeline.retimed(FrameRate::new(24, 1)).unwrap().frames, 192);
    // 8 s at 29.97 fps rounds to 240 frames.
    assert_eq!(timeline.retimed(NTSC).unwrap().frames, 240);
    // One frame at 60 fps retimed to 1 fps keeps at least one frame.
    let short = Timeline::new(FrameRate::new(60, 1), 1).unwrap();
    assert_eq!(short.retimed(FrameRate::new(1, 1)).unwrap().frames, 1);
    // Count overflow is an error rather than a wrap.
    let long = Timeline::new(FrameRate::new(1, 1), u32::MAX).unwrap();
    assert!(long.retimed(FrameRate::new(2, 1)).is_err());
    assert!(timeline.retimed(FrameRate::new(24, 0)).is_err());
}
