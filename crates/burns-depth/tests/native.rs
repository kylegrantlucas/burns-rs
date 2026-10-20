#![cfg(all(feature = "native", not(target_arch = "wasm32")))]

use std::{
    fs,
    io::{Read, Write},
    net::TcpListener,
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
    thread::{self, JoinHandle},
};

use burns::{DepthMap, Size, SourceImage};
use burns_depth::{DepthError, DepthProgress, DepthService, ModelSource, ensure_model_file};
use sha2::{Digest, Sha256};

/// One canned HTTP response: the declared Content-Length (`None` delimits the body by closing
/// the connection) and the body bytes actually sent before closing.
struct Reply {
    declared: Option<usize>,
    body: Vec<u8>,
}

/// Serves `replies` in order, one per connection, from a local socket.
fn serve(replies: Vec<Reply>) -> (String, JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}/model.onnx", listener.local_addr().unwrap());
    let handle = thread::spawn(move || {
        for reply in replies {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = Vec::new();
            let mut byte = [0u8; 1];
            while !request.ends_with(b"\r\n\r\n") {
                if stream.read(&mut byte).unwrap() == 0 {
                    break;
                }
                request.push(byte[0]);
            }
            let length = reply
                .declared
                .map(|declared| format!("Content-Length: {declared}\r\n"))
                .unwrap_or_default();
            let header = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\n\
                 {length}Connection: close\r\n\r\n"
            );
            // The client may hang up early (length check or cancellation); ignore that.
            let _ = stream
                .write_all(header.as_bytes())
                .and_then(|()| stream.write_all(&reply.body))
                .and_then(|()| stream.flush());
        }
    });
    (url, handle)
}

fn payload() -> Vec<u8> {
    // Several download chunks long.
    (0..900_000u32).map(|i| (i * 31 % 251) as u8).collect()
}

fn source_for(url: &str, bytes: &[u8]) -> ModelSource {
    ModelSource {
        url: url.to_owned(),
        bytes: bytes.len() as u64,
        sha256: hex::encode(Sha256::digest(bytes)),
    }
}

fn entries(dir: &Path) -> Vec<String> {
    match fs::read_dir(dir) {
        Ok(read) => read
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect(),
        Err(_) => Vec::new(),
    }
}

#[test]
fn verified_download_is_published_and_reused_without_network() {
    let body = payload();
    let (url, server) = serve(vec![Reply {
        declared: Some(body.len()),
        body: body.clone(),
    }]);
    let dir = tempfile::tempdir().unwrap();
    let source = source_for(&url, &body);
    let never = AtomicBool::new(false);
    let mut events = Vec::new();
    let path = ensure_model_file(dir.path(), &source, &never, |e| events.push(e)).unwrap();
    server.join().unwrap();
    assert_eq!(path, dir.path().join(source.file_name()));
    assert_eq!(fs::read(&path).unwrap(), body);
    assert_eq!(entries(dir.path()), vec![source.file_name()]);
    assert_eq!(
        events.last(),
        Some(&DepthProgress::Downloading {
            received: body.len() as u64,
            total: body.len() as u64
        })
    );

    // The server is gone; the existing file is verified and trusted.
    let mut events = Vec::new();
    let again = ensure_model_file(dir.path(), &source, &never, |e| events.push(e)).unwrap();
    assert_eq!(again, path);
    assert_eq!(events, vec![DepthProgress::Verifying]);
}

#[test]
fn wrong_hash_is_rejected_and_nothing_is_published() {
    let body = payload();
    let mut tampered = body.clone();
    tampered[123_456] ^= 1;
    let (url, server) = serve(vec![Reply {
        declared: Some(tampered.len()),
        body: tampered,
    }]);
    let dir = tempfile::tempdir().unwrap();
    let source = source_for(&url, &body);
    let err = ensure_model_file(dir.path(), &source, &AtomicBool::new(false), |_| {}).unwrap_err();
    server.join().unwrap();
    assert!(matches!(err, DepthError::ModelHash { .. }), "{err}");
    assert!(entries(dir.path()).is_empty(), "{:?}", entries(dir.path()));
}

#[test]
fn wrong_length_is_rejected_and_nothing_is_published() {
    let body = payload();
    let short = body[..body.len() - 10].to_vec();
    let long = [body.as_slice(), b"extra"].concat();
    let (url, server) = serve(vec![
        Reply {
            declared: Some(short.len()),
            body: short,
        },
        // No declared length; the body runs past the model's length.
        Reply {
            declared: None,
            body: long,
        },
    ]);
    let dir = tempfile::tempdir().unwrap();
    let source = source_for(&url, &body);
    let cancel = AtomicBool::new(false);
    let err = ensure_model_file(dir.path(), &source, &cancel, |_| {}).unwrap_err();
    assert!(
        matches!(err, DepthError::ModelLength { actual, .. } if actual == body.len() as u64 - 10),
        "{err}"
    );
    let err = ensure_model_file(dir.path(), &source, &cancel, |_| {}).unwrap_err();
    server.join().unwrap();
    assert!(matches!(err, DepthError::ModelTooLong { .. }), "{err}");
    assert!(entries(dir.path()).is_empty(), "{:?}", entries(dir.path()));
}

#[test]
fn interrupted_transfer_cannot_become_a_cache_hit() {
    let body = payload();
    let (url, server) = serve(vec![
        Reply {
            declared: Some(body.len()),
            body: body[..body.len() / 2].to_vec(),
        },
        Reply {
            declared: Some(body.len()),
            body: body.clone(),
        },
    ]);
    let dir = tempfile::tempdir().unwrap();
    let source = source_for(&url, &body);
    let cancel = AtomicBool::new(false);
    let err = ensure_model_file(dir.path(), &source, &cancel, |_| {}).unwrap_err();
    assert!(
        matches!(
            err,
            DepthError::Download { .. } | DepthError::ModelLength { .. }
        ),
        "{err}"
    );
    assert!(entries(dir.path()).is_empty(), "{:?}", entries(dir.path()));

    // The next attempt downloads again rather than trusting leftovers.
    let mut events = Vec::new();
    let path = ensure_model_file(dir.path(), &source, &cancel, |e| events.push(e)).unwrap();
    server.join().unwrap();
    assert_eq!(fs::read(path).unwrap(), body);
    assert!(!events.contains(&DepthProgress::Verifying));
}

#[test]
fn cancelling_mid_transfer_publishes_nothing() {
    let body = payload();
    let (url, server) = serve(vec![Reply {
        declared: Some(body.len()),
        body: body.clone(),
    }]);
    let dir = tempfile::tempdir().unwrap();
    let source = source_for(&url, &body);
    let cancel = AtomicBool::new(false);
    let mut received_when_cancelled = 0;
    let err = ensure_model_file(dir.path(), &source, &cancel, |event| {
        if let DepthProgress::Downloading { received, .. } = event
            && received > 0
            && !cancel.load(Ordering::Relaxed)
        {
            received_when_cancelled = received;
            cancel.store(true, Ordering::Relaxed);
        }
    })
    .unwrap_err();
    server.join().unwrap();
    assert!(matches!(err, DepthError::Cancelled), "{err}");
    assert!(received_when_cancelled < body.len() as u64);
    assert!(entries(dir.path()).is_empty(), "{:?}", entries(dir.path()));
}

#[test]
fn corrupt_existing_model_is_replaced_not_trusted() {
    let body = payload();
    let (url, server) = serve(vec![Reply {
        declared: Some(body.len()),
        body: body.clone(),
    }]);
    let dir = tempfile::tempdir().unwrap();
    let source = source_for(&url, &body);
    let dest = dir.path().join(source.file_name());
    let mut corrupt = body.clone();
    corrupt[0] ^= 0xff;
    fs::write(&dest, corrupt).unwrap();
    let mut events = Vec::new();
    let path = ensure_model_file(dir.path(), &source, &AtomicBool::new(false), |e| {
        events.push(e)
    })
    .unwrap();
    server.join().unwrap();
    assert_eq!(events[0], DepthProgress::Verifying);
    assert!(
        events
            .iter()
            .any(|e| matches!(e, DepthProgress::Downloading { .. }))
    );
    assert_eq!(fs::read(path).unwrap(), body);
}

fn image_16x9() -> SourceImage {
    let (width, height) = (32u32, 18u32);
    let pixels = (0..width * height)
        .flat_map(|i| [(i % 256) as u8, (i / 3 % 256) as u8, 7, 255])
        .collect();
    SourceImage::from_rgba8(Size { width, height }, pixels).unwrap()
}

fn gradient(width: u32, height: u32) -> DepthMap {
    let count = width * height;
    let values = (0..count)
        .map(|i| (f64::from(i) / f64::from(count - 1) * 65535.0).round() as f32 / 65535.0)
        .collect();
    DepthMap::from_f32(Size { width, height }, values).unwrap()
}

#[test]
fn cached_map_is_found_without_model_runtime_or_network() {
    let root = tempfile::tempdir().unwrap();
    let mut service = DepthService::with_cache_root(root.path());
    let image = image_16x9();
    assert!(service.cached(&image).unwrap().is_none());

    let map = gradient(924, 518);
    let path = service.depth_cache_path(&image);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(&path, map.encode_png().unwrap()).unwrap();
    assert_eq!(service.cached(&image).unwrap(), Some(map.clone()));

    // generate() returns the cached map first: no model download or runtime load.
    let mut events = Vec::new();
    let generated = service
        .generate(&image, &AtomicBool::new(false), |e| events.push(e))
        .unwrap();
    assert_eq!(generated, map);
    assert_eq!(events, vec![DepthProgress::CacheHit]);
    assert!(!root.path().join("models").exists());
}

#[test]
fn cached_map_with_wrong_geometry_is_reported_corrupt() {
    let root = tempfile::tempdir().unwrap();
    let service = DepthService::with_cache_root(root.path());
    let image = image_16x9();
    let path = service.depth_cache_path(&image);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(&path, gradient(518, 518).encode_png().unwrap()).unwrap();
    assert!(matches!(
        service.cached(&image),
        Err(DepthError::CacheCorrupt { .. })
    ));
    fs::write(&path, b"not a png").unwrap();
    assert!(matches!(
        service.cached(&image),
        Err(DepthError::CacheCorrupt { .. })
    ));
}

#[test]
fn cancelled_generation_does_nothing() {
    let root = tempfile::tempdir().unwrap();
    let mut service = DepthService::with_cache_root(root.path());
    let err = service
        .generate(&image_16x9(), &AtomicBool::new(true), |_| {})
        .unwrap_err();
    assert!(matches!(err, DepthError::Cancelled));
    assert!(entries(root.path()).is_empty());
}

#[test]
fn cancellation_interrupts_a_stalled_response_body() {
    use std::{sync::mpsc, time::Duration};

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}/model.onnx", listener.local_addr().unwrap());
    let (release_tx, release_rx) = mpsc::channel();
    let server = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut request = Vec::new();
        let mut byte = [0; 1];
        while !request.ends_with(b"\r\n\r\n") {
            assert_eq!(stream.read(&mut byte).unwrap(), 1);
            request.push(byte[0]);
        }
        stream
            .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 900000\r\nConnection: close\r\n\r\n")
            .unwrap();
        stream.flush().unwrap();
        // Deliberately send no body until the test releases the owned connection.
        release_rx.recv().unwrap();
    });
    let dir = tempfile::tempdir().unwrap();
    let source = source_for(&url, &payload());
    let cancel = AtomicBool::new(false);
    let (started_tx, started_rx) = mpsc::channel();
    let (done_tx, done_rx) = mpsc::channel();
    let timely = thread::scope(|scope| {
        scope.spawn(|| {
            let result = ensure_model_file(dir.path(), &source, &cancel, |event| {
                if matches!(event, DepthProgress::Downloading { received: 0, .. }) {
                    started_tx.send(()).unwrap();
                }
            });
            done_tx.send(result).unwrap();
        });
        started_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        // Let the client enter its body read before cancelling.
        thread::sleep(Duration::from_millis(100));
        cancel.store(true, Ordering::Relaxed);
        let result = done_rx.recv_timeout(Duration::from_secs(2));
        release_tx.send(()).unwrap();
        server.join().unwrap();
        result
    });
    assert!(
        matches!(timely, Ok(Err(DepthError::Cancelled))),
        "{timely:?}"
    );
    assert!(entries(dir.path()).is_empty());
}
