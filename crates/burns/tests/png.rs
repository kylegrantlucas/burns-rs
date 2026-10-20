use std::io::Cursor;

use burns::{PngError, Size, write_rgba8_png};

#[test]
fn png_rows_ignore_padding_and_preserve_rgba_with_an_srgb_tag() {
    let rows = [
        240, 10, 30, 255, 20, 230, 40, 255, 199, 198, 197, 196, 50, 60, 220, 255, 210, 200, 90,
        255, 195, 194, 193, 192,
    ];
    let mut encoded = Vec::new();
    write_rgba8_png(&mut encoded, Size::new(2, 2), &rows, 12, || false).unwrap();
    let mut reader = png::Decoder::new(Cursor::new(encoded)).read_info().unwrap();
    assert_eq!(
        reader.info().srgb,
        Some(png::SrgbRenderingIntent::Perceptual)
    );
    let mut rgba = [0; 16];
    let frame = reader.next_frame(&mut rgba).unwrap();
    assert_eq!(
        (frame.width, frame.height, frame.color_type),
        (2, 2, png::ColorType::Rgba)
    );
    assert_eq!(
        rgba,
        [
            240, 10, 30, 255, 20, 230, 40, 255, 50, 60, 220, 255, 210, 200, 90, 255
        ]
    );
}

#[test]
fn png_layout_errors_and_prior_cancellation_do_not_touch_the_writer() {
    for (bytes, stride) in [(vec![0; 16], 7), (vec![0; 15], 8)] {
        let mut output = vec![91, 92];
        assert!(matches!(
            write_rgba8_png(&mut output, Size::new(2, 2), &bytes, stride, || false),
            Err(PngError::Layout { .. })
        ));
        assert_eq!(output, [91, 92]);
    }
    let mut output = vec![91, 92];
    assert!(matches!(
        write_rgba8_png(&mut output, Size::new(1, 1), &[1, 2, 3, 255], 4, || true),
        Err(PngError::Cancelled)
    ));
    assert_eq!(output, [91, 92]);
}
