import { dump as dumpYaml } from 'js-yaml'

import type { ConfigMode, JsonObject, JsonSchema, JsonValue } from './schemaRegistry'

const TOP_LEVEL_ORDER: Record<ConfigMode, string[]> = {
  train: ['seed', 'device', 'datasource', 'dataset', 'model', 'trainer', 'strategy'],
  infer: ['seed', 'device', 'datasource', 'dataset', 'model', 'inferencer', 'strategy'],
  analyze: ['seed', 'selectors', 'collectors', 'aggregators', 'reporters', 'plotters'],
}

const DISCRIMINATOR_KEYS = [
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

function cloneValue<T extends JsonValue | undefined>(value: T): T {
  if (value === undefined) {
    return value
  }
  return JSON.parse(JSON.stringify(value)) as T
}

function constValues(schema: JsonValue | undefined): JsonValue[] {
  if (!isObject(schema)) {
    return []
  }
  if ('const' in schema) {
    return [schema.const]
  }
  if (Array.isArray(schema.enum)) {
    return schema.enum
  }
  return []
}

function schemaTypeIncludes(schema: JsonValue | undefined, type: string) {
  if (!isObject(schema)) {
    return false
  }
  return schema.type === type || (Array.isArray(schema.type) && schema.type.includes(type))
}

function isNullSchema(schema: JsonValue | undefined) {
  if (!isObject(schema)) {
    return false
  }
  if (schema.type === 'null') {
    return true
  }
  if (Array.isArray(schema.type) && schema.type.length === 1 && schema.type[0] === 'null') {
    return true
  }
  if ('const' in schema && schema.const === null) {
    return true
  }
  return Array.isArray(schema.enum) && schema.enum.length === 1 && schema.enum[0] === null
}

function schemaAcceptsNull(schema: JsonSchema) {
  if (schemaTypeIncludes(schema, 'null')) {
    return true
  }
  if (Array.isArray(schema.enum) && schema.enum.includes(null)) {
    return true
  }
  const options = Array.isArray(schema.oneOf) ? schema.oneOf : Array.isArray(schema.anyOf) ? schema.anyOf : undefined
  return options?.some(isNullSchema) ?? false
}

function isFixedValueSchema(schema: JsonValue | undefined) {
  return constValues(schema).length === 1
}

function scoreSubschema(schema: JsonSchema, data: JsonValue | undefined): number {
  if (!isObject(data)) {
    return 0
  }

  if (Array.isArray(schema.oneOf)) {
    return Math.max(...schema.oneOf.map((option) => scoreSubschema(option as JsonSchema, data)))
  }
  if (Array.isArray(schema.anyOf)) {
    return Math.max(...schema.anyOf.map((option) => scoreSubschema(option as JsonSchema, data)))
  }

  if (!isObject(schema.properties)) {
    return 0
  }

  let score = 0
  for (const [key, propSchema] of Object.entries(schema.properties)) {
    const acceptedValues = constValues(propSchema)
    if (acceptedValues.length > 0 && key in data) {
      if (acceptedValues.includes(data[key])) {
        score += 100
      } else {
        return -1
      }
    } else if (key in data) {
      score += 1
    }
  }

  return score
}

function chooseSubschema(options: JsonValue[], data: JsonValue | undefined): JsonSchema | undefined {
  let bestOption: JsonSchema | undefined
  let bestScore = -1

  for (const option of options) {
    if (!isObject(option)) {
      continue
    }
    const score = scoreSubschema(option, data)
    if (score > bestScore) {
      bestOption = option
      bestScore = score
    }
  }

  return bestOption ?? (isObject(options[0]) ? options[0] : undefined)
}

function schemaDefault(schema: JsonSchema): JsonValue | undefined {
  if ('default' in schema) {
    return cloneValue(schema.default)
  }
  if ('const' in schema) {
    return cloneValue(schema.const)
  }
  if (schemaAcceptsNull(schema)) {
    return null
  }
  return undefined
}

function shouldMaterializeProperty(schema: JsonSchema, propertyName: string, propSchema: JsonSchema, source: JsonObject) {
  if (propertyName in source) {
    return true
  }
  if (Array.isArray(schema.required) && schema.required.includes(propertyName)) {
    return true
  }
  return schemaDefault(propSchema) !== undefined
}

export function materializeDefaults(schema: JsonSchema, data: JsonValue | undefined): JsonValue | undefined {
  let initialValue = data === undefined ? schemaDefault(schema) : cloneValue(data)

  if (Array.isArray(schema.oneOf)) {
    const selected = chooseSubschema(schema.oneOf, initialValue)
    return selected ? materializeDefaults(selected, initialValue) : initialValue
  }

  if (Array.isArray(schema.anyOf)) {
    const selected = chooseSubschema(schema.anyOf, initialValue)
    return selected ? materializeDefaults(selected, initialValue) : initialValue
  }

  if (Array.isArray(schema.allOf)) {
    initialValue = schema.allOf.reduce(
      (currentValue, subschema) => (isObject(subschema) ? materializeDefaults(subschema, currentValue) : currentValue),
      initialValue,
    )
  }

  if (isObject(schema.properties)) {
    const source = isObject(initialValue) ? initialValue : {}
    const output: JsonObject = { ...source }

    for (const [key, propSchema] of Object.entries(schema.properties)) {
      if (!isObject(propSchema)) {
        continue
      }
      if (!shouldMaterializeProperty(schema, key, propSchema, source)) {
        continue
      }
      const valueWithDefaults = materializeDefaults(propSchema, output[key])
      if (valueWithDefaults !== undefined) {
        output[key] = valueWithDefaults
      }
    }

    return output
  }

  if (Array.isArray(initialValue) && isObject(schema.items)) {
    return initialValue.map((item) => materializeDefaults(schema.items as JsonSchema, item) ?? item)
  }

  return initialValue
}

function orderBySchema(schema: JsonSchema, data: JsonValue | undefined): JsonValue | undefined {
  if (Array.isArray(schema.oneOf)) {
    const selected = chooseSubschema(schema.oneOf, data)
    return selected ? orderBySchema(selected, data) : data
  }

  if (Array.isArray(schema.anyOf)) {
    const selected = chooseSubschema(schema.anyOf, data)
    return selected ? orderBySchema(selected, data) : data
  }

  if (Array.isArray(data)) {
    return data.map((item) => {
      const orderedItem = isObject(schema.items) ? orderBySchema(schema.items as JsonSchema, item) : item
      return orderedItem === undefined ? null : orderedItem
    })
  }

  if (!isObject(data) || !isObject(schema.properties)) {
    return data
  }

  const ordered: JsonObject = {}
  for (const key of DISCRIMINATOR_KEYS) {
    if (key in data) {
      ordered[key] = data[key]
    }
  }

  for (const [key, propSchema] of Object.entries(schema.properties)) {
    if (key in data && !(key in ordered) && isFixedValueSchema(propSchema)) {
      ordered[key] = data[key]
    }
  }

  for (const [key, propSchema] of Object.entries(schema.properties)) {
    if (key in data && !(key in ordered)) {
      ordered[key] = isObject(propSchema) ? orderBySchema(propSchema, data[key]) ?? null : data[key]
    }
  }

  for (const [key, value] of Object.entries(data)) {
    if (!(key in ordered)) {
      ordered[key] = value
    }
  }

  return ordered
}

export function buildCompleteConfig(schema: JsonSchema, data: JsonValue | undefined): JsonObject {
  const complete = materializeDefaults(schema, data)
  return isObject(complete) ? complete : {}
}

export function detectConfigMode(config: JsonObject): ConfigMode {
  const analysisKeys = ['selectors', 'collectors', 'aggregators', 'reporters', 'plotters']
  if (analysisKeys.every((key) => key in config)) {
    return 'analyze'
  }
  if ('inferencer' in config && !('trainer' in config)) {
    return 'infer'
  }
  return 'train'
}

export function formatYamlConfig(config: JsonObject, schema: JsonSchema, mode: ConfigMode): string {
  const orderedConfig = orderBySchema(schema, config)
  const topLevelOrder = TOP_LEVEL_ORDER[mode]
  const source = isObject(orderedConfig) ? orderedConfig : config
  const keys = [...topLevelOrder, ...Object.keys(source).filter((key) => !topLevelOrder.includes(key))]
  const sections = keys
    .filter((key) => key in source)
    .map((key) => dumpYaml({ [key]: source[key] }, { indent: 2, lineWidth: 100, noRefs: true, sortKeys: false }).trimEnd())

  return `${sections.join('\n\n')}\n`
}