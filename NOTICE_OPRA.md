# OPRA Attribution And License Notes

This project includes support for consuming OPRA database content and generating FIR exports from OPRA EQ entries.

Relevant upstream project:

- OPRA repository: https://github.com/opra-project/OPRA
- Hosted mirror for non-commercial, open source, and personal use: http://opra.roonlabs.net/database_v1.jsonl

Important upstream licensing split:

- OPRA repository source code is published under the MIT License.
- OPRA dataset content is published under Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0).

Practical compliance steps for this project:

- Preserve attribution to the OPRA Project when showing or redistributing OPRA-derived presets or exports.
- Prefer the hosted mirror instead of hitting GitHub directly unless you truly need the newest unmirrored data.
- Generated OPRA export folders include a `NOTICE_OPRA.txt` file to carry attribution forward with each pack.
- Generated OPRA export folders also include `ATTRIBUTION_OPRA.json` with UI-ready title, subtitle, and credit strings for app display.

Suggested in-app wording:

- Title: use the product name.
- Subtitle: use `Vendor | target`.
- Short credit: `Preset from OPRA for <product name>`.
- Long credit: `Product metadata and EQ preset from OPRA. Vendor: <vendor>. Product: <product>. Target: <target>. Measurement: <measurement>.`
- Legal footer: `OPRA dataset attribution required. See NOTICE_OPRA.txt for license details.`

Commercial/distribution note:

- If you redistribute OPRA-derived data or artifacts generated from OPRA entries, review whether CC BY-SA 4.0 share-alike obligations apply to your distribution model and product packaging.
- This file is an engineering notice, not legal advice.
