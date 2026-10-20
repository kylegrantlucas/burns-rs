use burns::{preset, presets};
use std::{error::Error, path::PathBuf};

fn main() -> Result<(), Box<dyn Error>> {
    let directory = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("presets"));
    std::fs::create_dir_all(&directory)?;
    for descriptor in presets() {
        let json = preset(descriptor.name)?.to_json()?;
        std::fs::write(
            directory.join(format!("{}.json", descriptor.name)),
            format!("{json}\n"),
        )?;
    }
    println!(
        "Generated {} portable presets in {}",
        presets().len(),
        directory.display()
    );
    Ok(())
}
