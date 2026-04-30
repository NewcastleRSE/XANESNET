# XANESNET Config UI

Interactive React editor for XANESNET `train`, `infer`, and `analyze` YAML configuration files. The form is generated from the JSON Schema files in `src/schemas`, so defaults, allowed variants, and compatibility rules stay close to the Python configuration model.

The app is intended for two jobs:

- Give users a browsable reference for valid XANESNET configs.
- Generate complete YAML files for training, inference, and analysis workflows.

## Features

- Mode-specific forms for Train, Infer, and Analyze configs.
- YAML import with automatic mode detection.
- Infer checkpoint `signature.yaml` import that merges signature-provided dataset/model/strategy fields into the Infer form.
- Signature-origin highlighting while keeping signature-loaded fields editable.
- Live YAML preview with schema defaults materialized and top-level sections ordered for readability.
- Light/dark theme toggle stored in local browser state.

## Quick Start

Run from this directory:

```bash
npm install
npm run dev
```

The development server prints the local URL, usually `http://127.0.0.1:5173/` or the next free Vite port.

## Common Commands

```bash
npm run dev      # Start the Vite development server
npm run lint     # Run ESLint
npm run build    # Type-check and build production assets into dist/
npm run preview  # Preview the production build locally
```

## Project Layout

- `src/App.tsx` contains the React UI, RJSF custom fields/templates, YAML loading, signature loading, and form state handling.
- `src/configYaml.ts` materializes schema defaults and formats generated YAML.
- `src/schemaRegistry.ts` loads YAML schemas through Vite, resolves local `$ref`s, and decorates union option labels.
- `src/schemas/` contains the bundled JSON Schemas for Train, Infer, Analyze, and their component sections.
- `public/favicon.svg` is the app icon used by `index.html`.

## Schema Notes

Schemas are authored as YAML files and imported with Vite raw imports. Local `$ref`s are resolved in `schemaRegistry.ts`, so schema files can stay split by domain while the app receives dereferenced schema objects.

Infer configs are special: at runtime XANESNET merges user Infer YAML with a checkpoint signature before validation. The UI mirrors that by letting users load `signature.yaml` in Infer mode. User-supplied Infer YAML can stay as a checkpoint overlay without `dataset_type`, `model`, or `strategy`; a loaded signature supplies those fields when available.

When changing schemas, smoke-test at least these flows:

- Train default descriptor/MLP generation.
- Train graph dataset plus graph model branch switching.
- Infer overlay generation without a signature.
- Infer YAML loaded before a signature.
- Signature loaded before a YAML config.
- Clear after loading a signature.
- Analyze empty lists and a populated analysis example.

## Maintenance

Keep generated build output out of source review. `dist/` and `node_modules/` are ignored and can be recreated with `npm run build` and `npm install`.

The app intentionally does not depend on a component framework. Styling lives in `src/App.css` and `src/index.css`; form rendering is handled by `@rjsf/core` and `@rjsf/validator-ajv8`.
