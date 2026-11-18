# Documentation asset license sources

Original upstream texts for assets embedded by the pinned documentation themes:

- `tools/docs/licenses/vendor/LicenseRef-Font-Awesome.txt` (Font Awesome Free 7.2.0): https://raw.githubusercontent.com/FortAwesome/Font-Awesome/7.2.0/LICENSE.txt
- `tools/docs/licenses/vendor/MIT.txt` (Bootstrap 5.3.3): https://raw.githubusercontent.com/twbs/bootstrap/v5.3.3/LICENSE
- `tools/docs/licenses/vendor/LicenseRef-ClipboardJS.txt` (clipboard.js 2.0.8): https://raw.githubusercontent.com/zenorocha/clipboard.js/v2.0.8/LICENSE
- `tools/docs/licenses/vendor/CC-BY-4.0.txt`: https://raw.githubusercontent.com/creativecommons/creativecommons.org/main/docroot/legalcode/by_4.0.txt
  (official plain text for https://creativecommons.org/licenses/by/4.0/legalcode.txt).

The documentation retains upstream asset attribution and license comment files.

Matching copies under `LICENSES/` are the source inventory authorities required by REUSE.

## Nested assets

The pinned pydata-sphinx-theme 0.23.0 bundles Popper 2.11.8, compare-versions
5.0.3 and Webpack 5.105.1 runtime code. Their identities come from the
[upstream release lock](https://github.com/pydata/pydata-sphinx-theme/blob/v0.23.0/package-lock.json);
the distributed source maps identify the corresponding modules. Notices below
were copied verbatim from the exact npm archives and checked against that lock's
SHA-512 integrity values. Webpack is listed for its emitted runtime snippets,
not its complete build tool or build dependencies.

- `vendor/Popper-MIT.txt`: [@popperjs/core 2.11.8 archive](https://registry.npmjs.org/@popperjs/core/-/core-2.11.8.tgz), `package/LICENSE.md`.
- `vendor/Compare-Versions-MIT.txt`: [compare-versions 5.0.3 archive](https://registry.npmjs.org/compare-versions/-/compare-versions-5.0.3.tgz), `package/LICENSE`.
- `vendor/Webpack-MIT.txt`: [webpack 5.105.1 archive](https://registry.npmjs.org/webpack/-/webpack-5.105.1.tgz), `package/LICENSE`.
- `vendor/RFS-MIT.txt`: [RFS 10.0.0 archive](https://registry.npmjs.org/rfs/-/rfs-10.0.0.tgz), `package/LICENSE`. Its `scss.scss` matches the RFS source embedded in the distributed Bootstrap CSS source map.
- `vendor/Tabler-MIT.txt`: [original Tabler license](https://github.com/tabler/tabler-icons/blob/e2b97ef5e9fbfb3e1b1897d7a3696dbd15532441/LICENSE). The modified check and clipboard SVGs come from [sphinx-copybutton commit 22ec7cc](https://github.com/executablebooks/sphinx-copybutton/commit/22ec7cc9430a444b94faf6ca8d6fe3f662f1cf00), retained in 0.5.2. The upstream Tabler release number is not recorded, so its inventory version remains `NOASSERTION`; the copied notice is the original license revision in effect when those icons were imported.

`docs/licenses/components.json` in each built guide records actual shipped asset
paths and SHA-256 hashes; SDK and wheel inventories validate those bytes. Shared
bundles can appear under multiple components. These records document attribution,
not a Legal or OSRB approval.

The shipped `clipboard.min.js` byte-matches sphinx-copybutton 0.5.2's
[clipboard submodule revision](https://github.com/zenorocha/clipboard.js/tree/57345ab3ce1b3526d736e1b391257d7221854513).
Its [lock](https://github.com/zenorocha/clipboard.js/blob/57345ab3ce1b3526d736e1b391257d7221854513/package-lock.json)
identifies good-listener 1.2.2, select 1.1.2, delegate 3.2.0, tiny-emitter 2.1.0
and Webpack runtime 5.15.0. The first three declare MIT © Zeno Rocha in their
original READMEs and share the retained clipboard.js terms. `vendor/Tiny-Emitter-MIT.txt`
is the verbatim `package/LICENSE` from the integrity-verified
[tiny-emitter 2.1.0 archive](https://registry.npmjs.org/tiny-emitter/-/tiny-emitter-2.1.0.tgz).
Webpack 5.15.0 and 5.105.1 contain byte-identical original license files; one
retained `vendor/Webpack-MIT.txt` covers both runtime copies.
