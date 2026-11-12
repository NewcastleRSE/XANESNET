import assert from 'node:assert/strict'
import { after, test } from 'node:test'
import { fileURLToPath } from 'node:url'
import { createServer } from 'vite'

const root = fileURLToPath(new URL('..', import.meta.url))
const server = await createServer({
  configFile: false,
  root,
  server: { middlewareMode: true },
  appType: 'custom',
  logLevel: 'silent',
})

const [configYaml, schemaRegistry] = await Promise.all([
  server.ssrLoadModule('/src/configYaml.ts'),
  server.ssrLoadModule('/src/schemaRegistry.ts'),
])

after(async () => {
  await server.close()
})

test('detects partial Analyze configs', () => {
  const schemas = schemaRegistry.SCHEMAS_BY_MODE
  assert.equal(configYaml.detectConfigMode({ custom_modules: [] }, schemas), 'analyze')
  assert.equal(configYaml.detectConfigMode({ preload: true }, schemas), 'analyze')
  assert.equal(configYaml.detectConfigMode({ inferencer: {} }, schemas), 'infer')
  assert.equal(configYaml.detectConfigMode({ trainer: {} }, schemas), 'train')
})

test('keeps checkpoint-owned fields out of an Infer overlay', () => {
  const schema = configYaml.createInferFormSchema(schemaRegistry.SCHEMAS_BY_MODE.infer)
  const config = configYaml.buildCompleteConfig(schema, {
    datasource: { datasource_type: 'pmgjson', json_path: './data' },
    dataset: { root: './processed' },
    strategy: { strategy_type: 'single' },
    inferencer: { inferencer_type: 'basic' },
  })

  assert.deepEqual(config.dataset, { root: './processed' })
  assert.equal('model' in config, false)
  assert.equal('encodings' in config, false)
  assert.equal('default' in schema.properties.encodings, false)

  const withExplicitDefaults = configYaml.buildCompleteConfig(schema, {
    dataset: { root: './processed', descriptors: [{ descriptor_type: 'wacsf' }] },
    encodings: [{ encoding_type: 'subtract_average', average: [0.1] }],
  })
  assert.equal(withExplicitDefaults.dataset.descriptors[0].r_min, 1)
  assert.equal(withExplicitDefaults.encodings[0].per_element, false)

  const fullDatasetConfig = configYaml.buildCompleteConfig(schema, {
    dataset: { dataset_type: 'descriptor' },
  })
  assert.equal(fullDatasetConfig.dataset.descriptors[0].descriptor_type, 'wacsf')
})

test('materializes explicit nullable defaults without inventing nulls', () => {
  const config = configYaml.buildCompleteConfig(
    {
      type: 'object',
      properties: {
        optional: { type: ['string', 'null'] },
        explicitDefault: { type: ['string', 'null'], default: null },
      },
    },
    {},
  )

  assert.deepEqual(config, { explicitDefault: null })
})