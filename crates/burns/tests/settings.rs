use burns::presets::{preset, presets};
use burns::settings::{
    CameraKind, CameraMotion, ConfigError, DepthAdjustments, DepthMotion, MotionProject,
    SettingValue, ShakeProfile, ShakeSettings, Size, View, WarpKind, WarpSettings,
};
use burns::timeline::{FrameRate, Timeline};
use serde_json::{Value, json};

fn project_value(project: &MotionProject) -> Value {
    serde_json::from_str(&project.to_json().unwrap()).unwrap()
}

fn parse(value: &Value) -> Result<MotionProject, ConfigError> {
    MotionProject::from_json(&value.to_string())
}

fn with_frames(mut project: MotionProject, frames: u32) -> MotionProject {
    project.timeline.frames = frames;
    project
}

/// Every leaf path of a JSON object tree, as JSON pointers.
fn leaf_pointers(value: &Value, prefix: String, out: &mut Vec<String>) {
    match value {
        Value::Object(map) => {
            for (key, child) in map {
                leaf_pointers(child, format!("{prefix}/{key}"), out);
            }
        }
        _ => out.push(prefix),
    }
}

fn remove_pointer(value: &mut Value, pointer: &str) {
    let (parent, key) = pointer.rsplit_once('/').unwrap();
    value
        .pointer_mut(parent)
        .unwrap()
        .as_object_mut()
        .unwrap()
        .remove(key)
        .unwrap();
}

#[test]
fn saved_project_round_trips_with_every_block() {
    let project = MotionProject {
        output: Size::new(1280, 720),
        timeline: Timeline::from_duration(FrameRate::new(30000, 1001), 8.0).unwrap(),
        seed: 7,
        depth: DepthAdjustments { invert: true },
        camera: CameraMotion::Depth {
            motion: DepthMotion::DollyOut,
            height: 0.5,
            steady: 0.25,
            focus: 0.75,
            zoom: 0.9,
            amount: 0.6,
            cycles: 3,
        },
        shake: ShakeSettings {
            enabled: true,
            profile: ShakeProfile::Vibrate,
            amount: 1.5,
            speed: 0.5,
        },
        warp: WarpSettings::HeatShimmer {
            enabled: true,
            amplitude: 0.004,
            spatial_scale: 2.0,
            cycles: 3,
        },
        ..MotionProject::default()
    };

    let value = project_value(&project);
    assert_eq!(value["depth"]["invert"], json!(true));
    assert_eq!(value["camera"]["type"], json!("depth"));
    assert_eq!(value["camera"]["motion"], json!("dolly-out"));
    assert_eq!(value["shake"]["profile"], json!("vibrate"));
    assert_eq!(value["warp"]["type"], json!("heat-shimmer"));
    assert_eq!(value["warp"]["spatial_scale"], json!(2.0));
    assert_eq!(value["timeline"]["fps"], json!({"num": 30000, "den": 1001}));

    let loaded = MotionProject::from_json(&project.to_json().unwrap()).unwrap();
    assert_eq!(loaded, project);
}

#[test]
fn unknown_fields_are_rejected_at_every_level() {
    let still = MotionProject {
        camera: CameraMotion::Still {
            view: View::centered(1.0),
        },
        ..MotionProject::default()
    };
    let base = project_value(&still);
    for parent in [
        "",
        "/output",
        "/timeline",
        "/timeline/fps",
        "/camera",
        "/camera/view",
        "/depth",
        "/shake",
        "/warp",
    ] {
        let mut value = base.clone();
        value
            .pointer_mut(parent)
            .unwrap()
            .as_object_mut()
            .unwrap()
            .insert("bogus".to_owned(), json!(1));
        let error = parse(&value).unwrap_err();
        assert!(
            matches!(error, ConfigError::Json { .. }),
            "{parent}: {error}"
        );
        assert!(error.to_string().contains("bogus"), "{parent}: {error}");
    }
}

#[test]
fn every_field_is_mandatory_including_disabled_blocks() {
    // Default shake and warp are disabled; their fields are still required.
    for project in [
        MotionProject::default(),
        preset("figure-eight").unwrap(),
        preset("orbital").unwrap(),
        preset("heat-shimmer").unwrap(),
    ] {
        let base = project_value(&project);
        let mut pointers = Vec::new();
        leaf_pointers(&base, String::new(), &mut pointers);
        assert!(pointers.contains(&"/shake/speed".to_owned()));
        assert!(pointers.contains(&"/warp/enabled".to_owned()));
        assert!(pointers.contains(&"/camera/type".to_owned()));
        for pointer in pointers {
            let mut value = base.clone();
            remove_pointer(&mut value, &pointer);
            let error = parse(&value).unwrap_err();
            assert!(
                matches!(error, ConfigError::Json { .. }),
                "{pointer}: {error}"
            );
        }
    }
}

#[test]
fn unknown_enum_values_and_wrong_types_are_rejected() {
    let base = project_value(&MotionProject::default());
    for (pointer, replacement) in [
        ("/camera/type", json!("spin")),
        ("/camera/easing", json!("bounce")),
        ("/shake/profile", json!("wobble")),
        ("/warp/type", json!("rain")),
        ("/warp/cycles", json!(-1)),
        ("/warp/cycles", json!(1.5)),
        ("/shake/enabled", json!("yes")),
        ("/seed", json!(4_294_967_296u64)),
    ] {
        let mut value = base.clone();
        *value.pointer_mut(pointer).unwrap() = replacement;
        let error = parse(&value).unwrap_err();
        assert!(
            matches!(error, ConfigError::Json { .. }),
            "{pointer}: {error}"
        );
    }

    let mut value = base.clone();
    value["shake"]["speed"] = json!("fast");
    assert_eq!(parse(&value).unwrap_err().field(), "shake.speed");

    let trailing = format!("{} {{}}", MotionProject::default().to_json().unwrap());
    assert!(MotionProject::from_json(&trailing).is_err());
}

#[test]
fn out_of_range_and_nonfinite_values_are_rejected_with_their_field() {
    let base = project_value(&MotionProject::default());
    for (pointer, replacement, field) in [
        ("/camera/start/zoom", json!(0.5), "camera.start.zoom"),
        ("/camera/end/x", json!(1.01), "camera.end.x"),
        (
            "/camera/end/roll_degrees",
            json!(-46.0),
            "camera.end.roll_degrees",
        ),
        // Disabled blocks must still be in range.
        ("/shake/speed", json!(0.0), "shake.speed"),
        ("/shake/amount", json!(2.5), "shake.amount"),
        ("/warp/amplitude", json!(0.5), "warp.amplitude"),
        ("/warp/center", json!([0.5, -0.1]), "warp.center.1"),
        ("/warp/cycles", json!(0), "warp.cycles"),
        ("/warp/cycles", json!(33), "warp.cycles"),
        ("/output/width", json!(0), "output.width"),
        ("/timeline/frames", json!(0), "timeline.frames"),
        ("/timeline/fps/den", json!(0), "timeline.fps.den"),
        ("/version", json!(2), "version"),
        // Overflows f32 to infinity.
        ("/camera/start/zoom", json!(1e39), "camera.start.zoom"),
    ] {
        let mut value = base.clone();
        *value.pointer_mut(pointer).unwrap() = replacement;
        let error = parse(&value).unwrap_err();
        assert_eq!(error.field(), field, "{pointer}: {error}");
    }

    let project = MotionProject {
        camera: CameraMotion::Still {
            view: View::new(f32::NAN, 0.5, 0.5, 0.0),
        },
        ..MotionProject::default()
    };
    assert_eq!(project.validate().unwrap_err().field(), "camera.view.zoom");
    assert!(project.to_json().is_err());

    let mut project = MotionProject::default();
    let before = project.clone();
    assert!(
        project
            .set_value("shake.amount", SettingValue::Float(f32::NAN))
            .is_err()
    );
    for raw in ["inf", "-inf", "NaN", "abc", ""] {
        assert!(project.set_by_id("shake.amount", raw).is_err(), "{raw}");
    }
    assert_eq!(project, before);
}

#[test]
fn failed_edits_leave_the_project_unchanged() {
    let mut project = MotionProject::default();
    project.set_by_id("seed", "99").unwrap();
    let before = project.clone();

    let errors = [
        project.set_by_id("camera.start.zoom", "20"),
        project.set_by_id("camera.start.zoom", "big"),
        project.set_by_id("camera.type", "spin"),
        project.set_by_id("camera.type", "6"),
        project.set_value("camera.type", SettingValue::Float(1.0)),
        project.set_by_id("seed", "-1"),
        project.set_by_id("seed", "4294967296"),
        project.set_by_id("shake.enabled", "yes"),
        project.set_output(Size::new(0, 1080)),
        project.set_duration(f64::NAN),
        project.set_frames(0),
        project.set_fps(FrameRate::new(0, 1)),
        project.apply_preset("nope"),
    ];
    for error in errors {
        assert!(error.is_err());
    }
    assert_eq!(project, before);

    assert!(matches!(
        project.set_by_id("camera.bogus", "1"),
        Err(ConfigError::UnknownSetting { .. })
    ));
    // Fields of inactive variants are not addressable.
    assert!(matches!(
        project.set_by_id("camera.cycles", "2"),
        Err(ConfigError::InactiveSetting { .. })
    ));
    assert!(matches!(
        project.get_by_id("camera.view.zoom"),
        Err(ConfigError::InactiveSetting { .. })
    ));
    assert!(matches!(
        project.set_by_id("warp.spatial_scale", "1"),
        Err(ConfigError::InactiveSetting { .. })
    ));
    assert_eq!(project, before);
}

#[test]
fn periodic_motion_must_stay_below_nyquist() {
    // Breathing: cycles / duration < fps / 2  <=>  2 * cycles < frames.
    let mut project = with_frames(preset("breathing").unwrap(), 32);
    project.set_by_id("camera.cycles", "15").unwrap();
    let before = project.clone();
    let error = project.set_by_id("camera.cycles", "16").unwrap_err();
    assert_eq!(error.field(), "camera.cycles");
    let message = error.to_string();
    assert!(message.contains("15.000 Hz"), "{message}");
    assert!(message.contains("increase fps"), "{message}");
    assert_eq!(project, before);
    // Shortening the timeline is checked too.
    assert_eq!(project.set_frames(30).unwrap_err().field(), "camera.cycles");
    assert_eq!(project, before);
    // Without amplitude there is no periodic contribution.
    project.set_by_id("camera.amplitude", "0").unwrap();
    project.set_by_id("camera.cycles", "16").unwrap();

    // Figure eight runs at twice its cycle rate.
    assert!(
        with_frames(preset("figure-eight").unwrap(), 4)
            .validate()
            .is_err()
    );
    assert!(
        with_frames(preset("figure-eight").unwrap(), 5)
            .validate()
            .is_ok()
    );

    // Ripple at cycles 2 needs more than 4 frames; heat shimmer more than 8.
    assert!(
        with_frames(preset("ripple").unwrap(), 4)
            .validate()
            .is_err()
    );
    assert!(with_frames(preset("ripple").unwrap(), 5).validate().is_ok());
    assert!(
        with_frames(preset("heat-shimmer").unwrap(), 8)
            .validate()
            .is_err()
    );
    assert!(
        with_frames(preset("heat-shimmer").unwrap(), 9)
            .validate()
            .is_ok()
    );
    // Disabled warp is ignored temporally.
    let mut ripple = preset("ripple").unwrap();
    ripple.set_by_id("warp.enabled", "false").unwrap();
    assert!(with_frames(ripple, 1).validate().is_ok());

    // Dolly out ignores cycles; switching to a periodic dolly checks them.
    let mut dolly = with_frames(preset("dolly-out").unwrap(), 10);
    dolly.set_by_id("camera.cycles", "32").unwrap();
    let before = dolly.clone();
    assert_eq!(
        dolly
            .set_by_id("camera.motion", "dolly")
            .unwrap_err()
            .field(),
        "camera.cycles"
    );
    assert_eq!(dolly, before);
    // A zero-amount depth camera is not periodic.
    dolly.set_by_id("camera.amount", "0").unwrap();
    dolly.set_by_id("camera.motion", "orbital").unwrap();
}

#[test]
fn shake_frequency_scales_with_profile_and_speed() {
    // Vibrate tops out at 13.1 Hz: below 15 Hz at 30 fps, above 12 Hz at 24 fps.
    let mut project = preset("vibrate").unwrap();
    let before = project.clone();
    let error = project.set_fps(FrameRate::new(24, 1)).unwrap_err();
    assert_eq!(error.field(), "shake.speed");
    assert!(error.to_string().contains("13.100 Hz"), "{error}");
    assert_eq!(project, before);

    // 13.7 Hz * 1.1 = 15.07 Hz reaches 15 Hz at 30 fps.
    let mut tight = preset("vibrate-tight").unwrap();
    assert_eq!(
        tight.set_by_id("shake.speed", "1.1").unwrap_err().field(),
        "shake.speed"
    );
    tight.set_by_id("shake.speed", "1.09").unwrap();

    // Handheld shake at 3.7 Hz fits 24 fps even at speed 3.
    let mut shake = preset("shake").unwrap();
    shake.set_by_id("shake.speed", "3").unwrap();
    shake.set_fps(FrameRate::new(24, 1)).unwrap();
    assert_eq!(shake.timeline.frames, 192);

    // Disabled or zero-amount shake is ignored temporally.
    let mut disabled = preset("vibrate").unwrap();
    disabled.set_by_id("shake.enabled", "false").unwrap();
    disabled.set_fps(FrameRate::new(24, 1)).unwrap();
    let mut silent = preset("vibrate").unwrap();
    silent.set_by_id("shake.amount", "0").unwrap();
    silent.set_fps(FrameRate::new(24, 1)).unwrap();
}

#[test]
fn timing_edits_preserve_duration_unless_overridden() {
    let mut project = MotionProject::default();
    project.set_fps(FrameRate::new(60, 1)).unwrap();
    assert_eq!(project.timeline.frames, 480);
    project.set_duration(2.5).unwrap();
    assert_eq!(
        project.timeline,
        Timeline::new(FrameRate::new(60, 1), 150).unwrap()
    );
    project
        .set_timeline(Timeline::from_duration(FrameRate::new(24, 1), 3.0).unwrap())
        .unwrap();
    assert_eq!(project.timeline.frames, 72);
}

#[test]
fn camera_type_switch_installs_defaults_and_preserves_other_blocks() {
    let mut project = preset("ripple").unwrap();
    project.set_by_id("seed", "1234").unwrap();
    project.set_by_id("depth.invert", "true").unwrap();
    project.set_by_id("shake.amount", "0.5").unwrap();
    project.set_by_id("warp.cycles", "3").unwrap();
    let before = project.clone();

    project.set_by_id("camera.type", "breathing").unwrap();
    assert_eq!(
        project.camera,
        CameraMotion::default_for(CameraKind::Breathing)
    );
    assert_eq!(
        (project.seed, project.depth, project.shake, project.warp),
        (before.seed, before.depth, before.shake, before.warp)
    );
    assert_eq!(
        project.get_by_id("camera.type").unwrap(),
        SettingValue::Choice(3)
    );

    // Index selection, and reselecting the current type keeps edits.
    project.set_by_id("camera.type", "4").unwrap();
    assert_eq!(project.camera.kind(), CameraKind::FigureEight);
    project.set_by_id("camera.amplitude.1", "0.45").unwrap();
    project.set_by_id("camera.type", "figure-eight").unwrap();
    assert_eq!(
        project.get_by_id("camera.amplitude.1").unwrap(),
        SettingValue::Float(0.45)
    );
    assert_eq!(project.warp, before.warp);
}

#[test]
fn warp_type_switch_installs_defaults_and_keeps_enabled() {
    for enabled in [true, false] {
        let mut project = preset("ripple").unwrap();
        project
            .set_by_id("warp.enabled", if enabled { "true" } else { "false" })
            .unwrap();
        project.set_by_id("camera.view.zoom", "2").unwrap();
        let camera = project.camera;

        project.set_by_id("warp.type", "heat-shimmer").unwrap();
        assert_eq!(
            project.warp,
            WarpSettings::default_for(WarpKind::HeatShimmer, enabled)
        );
        assert_eq!(project.camera, camera);

        project.set_by_id("warp.type", "0").unwrap();
        assert_eq!(
            project.warp,
            WarpSettings::default_for(WarpKind::Ripple, enabled)
        );
    }
}

#[test]
fn depth_motion_switch_keeps_shared_depth_parameters() {
    let mut project = preset("orbital").unwrap();
    for (id, raw) in [
        ("camera.height", "0.5"),
        ("camera.steady", "0.1"),
        ("camera.focus", "0.9"),
        ("camera.zoom", "1.5"),
        ("camera.amount", "0.4"),
        ("camera.cycles", "3"),
    ] {
        project.set_by_id(id, raw).unwrap();
    }
    project.set_by_id("camera.motion", "circle").unwrap();
    assert_eq!(
        project.camera,
        CameraMotion::Depth {
            motion: DepthMotion::Circle,
            height: 0.5,
            steady: 0.1,
            focus: 0.9,
            zoom: 1.5,
            amount: 0.4,
            cycles: 3,
        }
    );
    // Depth zoom has its own range, distinct from figure-eight zoom.
    project.set_by_id("camera.zoom", "0.6").unwrap();
    project.set_by_id("camera.type", "figure-eight").unwrap();
    assert!(project.set_by_id("camera.zoom", "0.6").is_err());
}

#[test]
fn presets_replace_motion_and_preserve_project_settings() {
    let mut base = MotionProject::default();
    base.set_output(Size::new(640, 360)).unwrap();
    base.set_timeline(Timeline::new(FrameRate::new(60, 1), 300).unwrap())
        .unwrap();
    base.set_by_id("seed", "42").unwrap();
    base.set_by_id("depth.invert", "true").unwrap();
    // Leftover motion state that presets must replace.
    base.set_by_id("shake.enabled", "true").unwrap();
    base.set_by_id("warp.type", "heat-shimmer").unwrap();

    assert_eq!(presets().len(), 16);
    for descriptor in presets() {
        let mut project = base.clone();
        project.apply_preset(descriptor.name).unwrap();
        let reference = preset(descriptor.name).unwrap();
        assert_eq!(project.output, base.output, "{}", descriptor.name);
        assert_eq!(project.timeline, base.timeline, "{}", descriptor.name);
        assert_eq!(project.seed, 42);
        assert!(project.depth.invert);
        assert_eq!(
            (project.camera, project.shake, project.warp),
            (reference.camera, reference.shake, reference.warp),
            "{}",
            descriptor.name
        );
        assert_eq!(
            descriptor.requires_depth,
            project.camera.kind() == CameraKind::Depth,
            "{}",
            descriptor.name
        );
    }
    assert_eq!(preset("kenburns").unwrap(), MotionProject::default());
    assert!(matches!(
        preset("spin"),
        Err(ConfigError::UnknownPreset { .. })
    ));
}

#[test]
fn size_rejects_zero_and_unaddressable_dimensions() {
    assert_eq!(Size::new(2, 3).rgba_len().unwrap(), 24);
    assert_eq!(Size::new(0, 3).validate().unwrap_err().field(), "width");
    assert_eq!(Size::new(3, 0).validate().unwrap_err().field(), "height");
    assert!(Size::new(u32::MAX, u32::MAX).rgba_len().is_err());
}

#[test]
fn ordered_settings_batches_validate_the_final_state_and_roll_back_on_failure() {
    let mut project = preset("vibrate-tight").unwrap();
    project
        .set_many_by_id([("shake.speed", "2"), ("shake.enabled", "false")])
        .unwrap();
    assert_eq!(project.shake.speed, 2.0);
    assert!(!project.shake.enabled);
    assert_eq!(project.timeline.fps, FrameRate::new(30, 1));
    let before = project.clone();
    assert!(
        project
            .set_many_by_id([("shake.enabled", "true"), ("seed", "123"),])
            .is_err()
    );
    assert_eq!(project, before);
    project
        .set_many_by_id([("shake.enabled", "true"), ("shake.speed", "0.5")])
        .unwrap();
    assert!(project.shake.enabled);
    assert_eq!(project.shake.speed, 0.5);
    let before = project.clone();
    assert!(
        project
            .set_many_by_id([("camera.type", "breathing"), ("camera.base_zoom", "0"),])
            .is_err()
    );
    assert_eq!(project, before);
}

#[test]
fn impossible_displacement_is_rejected_before_a_project_can_be_committed_or_loaded() {
    let mut project = preset("vibrate").unwrap();
    project.set_by_id("shake.amount", "2").unwrap();
    let before = project.clone();
    let error = project.set_output(Size::new(2, 1000)).unwrap_err();
    assert_eq!(error.field(), "shake.amount");
    assert_eq!(project, before);
    let mut invalid = project;
    invalid.output = Size::new(2, 1000);
    let json = serde_json::to_string(&invalid).unwrap();
    assert_eq!(
        MotionProject::from_json(&json).unwrap_err().field(),
        "shake.amount"
    );
}
