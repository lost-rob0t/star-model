{
  description = "Star Model dataset, QLoRA, and Pi A/B evaluation lab";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";

  outputs =
    { self, nixpkgs, ... }:
    let
      systems = [ "x86_64-linux" ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f system);
    in
    {
      packages = forAllSystems (
        system:
        let
          pkgs = import nixpkgs { inherit system; };
          piPinned = pkgs.writeShellApplication {
            name = "pi-pinned";
            runtimeInputs = [ pkgs.nodejs_22 ];
            text = ''
              export PI_SKIP_VERSION_CHECK=1
              exec npm exec \
                --yes \
                --ignore-scripts \
                --package=@earendil-works/pi-coding-agent@0.83.0 \
                -- pi "$@"
            '';
          };
        in
        {
          pi-pinned = piPinned;
          default = piPinned;
        }
      );

      devShells = forAllSystems (
        system:
        let
          pkgs = import nixpkgs { inherit system; };
          piPinned = self.packages.${system}.pi-pinned;
        in
        {
          default = pkgs.mkShell {
            packages = with pkgs; [
              awscli2
              coreutils
              git
              jq
              nodejs_22
              python312
              rsync
              shellcheck
              swi-prolog
              uv
              zstd
              piPinned
            ];

            shellHook = ''
              export PI_SKIP_VERSION_CHECK=1
              export STAR_MODEL_LAB_ROOT="$PWD"
              echo "star-model lab: use scripts/lab.py, scripts/pi_eval.py, and pi-pinned"
            '';
          };
        }
      );
    };
}
