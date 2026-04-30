import { load as loadYaml } from 'js-yaml'

export type ConfigMode = 'train' | 'infer' | 'analyze'
export type JsonValue = null | boolean | number | string | JsonValue[] | JsonObject
export type JsonObject = { [key: string]: JsonValue }
export type JsonSchema = JsonObject

type RawSchemaMap = Record<string, JsonSchema>

const schemaModules = import.meta.glob('./schemas/**/*.schema.yaml', {
  eager: true,
  import: 'default',
  query: '?raw',
}) as Record<string, string>

const TYPE_KEYS = [
  'datasource_type',
  'dataset_type',
  'model_type',
  'trainer_type',
  'inferencer_type',
  'strategy_type',
  'selector_type',
  'collector_type',
  'aggregator_type',
  'reporter_type',
  'plotter_type',
  'descriptor_type',
  'loss_type',
  'regularizer_type',
  'early_stopper_type',
  'lr_scheduler_type',
  'name',
]

function isObject(value: unknown): value is JsonObject {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function normalizeModulePath(modulePath: string): string {
  return modulePath.replace(/^\.\/schemas\//, '')
}

function parseSchemas(): RawSchemaMap {
  const schemas: RawSchemaMap = {}

  for (const [modulePath, content] of Object.entries(schemaModules)) {
    const parsed = loadYaml(content)
    if (!isObject(parsed)) {
      throw new Error(`Schema file ${modulePath} does not contain a YAML object.`)
    }
    schemas[normalizeModulePath(modulePath)] = parsed
  }

  return schemas
}

const rawSchemas = parseSchemas()

function normalizeRelativePath(fromPath: string, targetPath: string): string {
  if (targetPath.startsWith('/')) {
    return targetPath.slice(1)
  }

  const parts = fromPath.split('/').slice(0, -1)
  for (const part of targetPath.split('/')) {
    if (!part || part === '.') {
      continue
    }
    if (part === '..') {
      parts.pop()
    } else {
      parts.push(part)
    }
  }
  return parts.join('/')
}

function decodePointerPart(value: string): string {
  return value.replace(/~1/g, '/').replace(/~0/g, '~')
}

function readPointer(schema: JsonValue, pointer: string): JsonValue {
  if (!pointer || pointer === '/') {
    return schema
  }

  const parts = pointer.replace(/^\//, '').split('/').map(decodePointerPart)
  let current: JsonValue = schema

  for (const part of parts) {
    if (!isObject(current) && !Array.isArray(current)) {
      throw new Error(`Cannot resolve JSON pointer segment "${part}".`)
    }
    current = Array.isArray(current) ? current[Number(part)] : current[part]
  }

  return current
}

function mergeSchemaObjects(base: JsonValue, override: JsonValue): JsonValue {
  if (!isObject(base) || !isObject(override)) {
    return override
  }
  return { ...base, ...override }
}

function resolveSchemaNode(value: JsonValue, currentPath: string, seenRefs = new Set<string>()): JsonValue {
  if (Array.isArray(value)) {
    return value.map((item) => resolveSchemaNode(item, currentPath, seenRefs))
  }

  if (!isObject(value)) {
    return value
  }

  const ref = typeof value.$ref === 'string' ? value.$ref : undefined
  if (ref) {
    const [filePart, pointerPart = ''] = ref.split('#')
    const targetPath = filePart ? normalizeRelativePath(currentPath, filePart) : currentPath
    const refKey = `${targetPath}#${pointerPart}`

    if (seenRefs.has(refKey)) {
      return { ...value }
    }

    const targetSchema = rawSchemas[targetPath]
    if (!targetSchema) {
      throw new Error(`Cannot resolve schema reference ${ref} from ${currentPath}.`)
    }

    const nextSeenRefs = new Set(seenRefs)
    nextSeenRefs.add(refKey)

    const target = readPointer(targetSchema, pointerPart)
    const resolvedTarget = resolveSchemaNode(target, targetPath, nextSeenRefs)
    const siblings = Object.fromEntries(Object.entries(value).filter(([key]) => key !== '$ref')) as JsonObject

    if (Object.keys(siblings).length === 0) {
      return resolvedTarget
    }

    return mergeSchemaObjects(resolvedTarget, resolveSchemaNode(siblings, currentPath, seenRefs))
  }

  const resolvedEntries = Object.entries(value).map(([key, entry]) => [
    key,
    resolveSchemaNode(entry, currentPath, seenRefs),
  ])
  return Object.fromEntries(resolvedEntries) as JsonObject
}

function removeSchemaBookkeeping(value: JsonValue): JsonValue {
  if (Array.isArray(value)) {
    return value.map(removeSchemaBookkeeping)
  }
  if (!isObject(value)) {
    return value
  }

  const cleaned: JsonObject = {}
  for (const [key, entry] of Object.entries(value)) {
    if (key === '$schema' || key === '$id') {
      continue
    }
    cleaned[key] = removeSchemaBookkeeping(entry)
  }
  return cleaned
}

function readConstLikeValue(schema: JsonValue): string | undefined {
  if (!isObject(schema)) {
    return undefined
  }
  if (typeof schema.const === 'string') {
    return schema.const
  }
  if (Array.isArray(schema.enum) && schema.enum.length > 0 && schema.enum.every((value) => typeof value === 'string')) {
    return schema.enum.join(' / ')
  }
  return undefined
}

function humanizeTypeKey(key: string): string {
  return key.replace(/_/g, ' ').replace(/ type$/, '')
}

function formatFixedProperty(key: string, value: string): string {
  return `${humanizeTypeKey(key)}: ${value}`
}

function readFixedPropertyValue(properties: JsonObject, key: string): string | undefined {
  const value = readConstLikeValue(properties[key])
  return value || undefined
}

function isDiscriminatorLikeKey(key: string): boolean {
  return key.endsWith('_type') || key === 'name'
}

function inferOptionTitle(option: JsonValue): string | undefined {
  if (!isObject(option) || !isObject(option.properties)) {
    return undefined
  }

  const fixedProperties = Object.keys(option.properties)
    .filter(isDiscriminatorLikeKey)
    .map((key) => ({ key, value: readFixedPropertyValue(option.properties as JsonObject, key) }))
    .filter((entry): entry is { key: string; value: string } => Boolean(entry.value))

  for (const typeKey of TYPE_KEYS) {
    const primary = fixedProperties.find((entry) => entry.key === typeKey)
    if (primary) {
      const secondary = fixedProperties.filter((entry) => entry.key !== primary.key)
      const titleParts = [primary, ...secondary].map((entry) => formatFixedProperty(entry.key, entry.value))
      return titleParts.join(' / ')
    }
  }

  if (fixedProperties.length > 0) {
    return fixedProperties.map((entry) => formatFixedProperty(entry.key, entry.value)).join(' / ')
  }

  return undefined
}

function decorateSchemaOptions(value: JsonValue): JsonValue {
  if (Array.isArray(value)) {
    return value.map(decorateSchemaOptions)
  }
  if (!isObject(value)) {
    return value
  }

  const decorated: JsonObject = {}
  for (const [key, entry] of Object.entries(value)) {
    if ((key === 'oneOf' || key === 'anyOf') && Array.isArray(entry)) {
      decorated[key] = entry.map((option) => {
        const decoratedOption = decorateSchemaOptions(option)
        if (!isObject(decoratedOption) || decoratedOption.title) {
          return decoratedOption
        }
        const title = inferOptionTitle(decoratedOption)
        return title ? { ...decoratedOption, title } : decoratedOption
      })
    } else {
      decorated[key] = decorateSchemaOptions(entry)
    }
  }

  return decorated
}

function buildSchema(entryPath: string): JsonSchema {
  const entry = rawSchemas[entryPath]
  if (!entry) {
    throw new Error(`Missing schema entry ${entryPath}.`)
  }

  return decorateSchemaOptions(removeSchemaBookkeeping(resolveSchemaNode(entry, entryPath))) as JsonSchema
}

const trainSchema = buildSchema('train.schema.yaml')
const inferSchema = buildSchema('infer.schema.yaml')
const analyzeSchema = buildSchema('analyze.schema.yaml')

export const CONFIG_SCHEMA_OPTIONS = [
  {
    id: 'train',
    label: 'Train',
    description: 'Build a training configuration with datasource, dataset, model, trainer, and strategy sections.',
    schema: trainSchema,
  },
  {
    id: 'infer',
    label: 'Infer',
    description: 'Build an inference configuration that can be merged with a checkpoint signature at runtime.',
    schema: inferSchema,
  },
  {
    id: 'analyze',
    label: 'Analyze',
    description: 'Build an analysis pipeline with selectors, collectors, aggregators, reporters, and plotters.',
    schema: analyzeSchema,
  },
] as const

export const SCHEMAS_BY_MODE: Record<ConfigMode, JsonSchema> = {
  train: trainSchema,
  infer: inferSchema,
  analyze: analyzeSchema,
}