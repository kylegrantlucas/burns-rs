{
  description = "burns-rs: deterministic single-image motion workbench";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { nixpkgs, rust-overlay, ... }:
    let
      systems = [ "aarch64-darwin" "x86_64-darwin" "x86_64-linux" "aarch64-linux" ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f (import nixpkgs {
        inherit system;
        overlays = [ rust-overlay.overlays.default ];
      }));
    in {
      devShells = forAllSystems (pkgs:
        let
          inherit (pkgs) lib stdenv;
          toolchain = pkgs.rust-bin.stable."1.97.0".default.override {
            targets = [ "wasm32-unknown-unknown" ];
          };
          linuxRuntime = with pkgs; [ libxkbcommon wayland libGL xorg.libX11 xorg.libXcursor xorg.libXi xorg.libXrandr ];
        in {
          default = pkgs.mkShell ({
            packages = [ toolchain ] ++ (with pkgs; [
              rust-analyzer just pkg-config lld nodejs_24 pnpm
              wasm-bindgen-cli_0_2_129 ffmpeg onnxruntime
            ]) ++ lib.optionals stdenv.hostPlatform.isDarwin [ pkgs.libiconv ]
              ++ lib.optionals stdenv.hostPlatform.isLinux linuxRuntime;
            ORT_DYLIB_PATH = "${lib.getLib pkgs.onnxruntime}/lib/libonnxruntime${stdenv.hostPlatform.extensions.sharedLibrary}";
          } // lib.optionalAttrs stdenv.hostPlatform.isLinux {
            LD_LIBRARY_PATH = lib.makeLibraryPath linuxRuntime;
          });
        });
    };
}
