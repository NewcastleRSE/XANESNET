import { createContext, useContext, useEffect, useMemo, useState } from 'react'
import Form, { type IChangeEvent } from '@rjsf/core'
import validator from '@rjsf/validator-ajv8'
import {
  buttonId,
  canExpand,
  type FieldProps,
  type IconButtonProps,
  type ObjectFieldTemplateProps,
  type UiSchema,
} from '@rjsf/utils'
import { load as loadYaml } from 'js-yaml'

import './App.css'
import { CONFIG_SCHEMA_OPTIONS, SCHEMAS_BY_MODE, type ConfigMode, type JsonObject, type JsonValue } from './schemaRegistry'
import { buildCompleteConfig, detectConfigMode, formatYamlConfig } from './configYaml'

type Status = {
  tone: 'idle' | 'success' | 'error'
  message: string
}

type ThemeMode = 'light' | 'dark'
type IconName = 'chevron-down' | 'chevron-up' | 'copy' | 'moon' | 'plus' | 'sun' | 'trash' | 'x'
type ObjectFieldProperty = ObjectFieldTemplateProps['properties'][number]
type AutoValueDefinition = {
  kind: 'number' | 'array'
  manualSchema: JsonObject
}
type NullableValueKind = 'array' | 'boolean' | 'enum' | 'integer' | 'mixed' | 'number' | 'string'
type NullableValueDefinition = {
  kind: NullableValueKind
  valueSchema: JsonObject
  enumValues?: JsonValue[]
  itemSchema?: JsonObject
}
type SignatureState = {
  fileName: string
  paths: string[]
  leafPaths: string[]
  schemaExtraLeafPaths: string[]
}
type SignatureContextValue = {
  fileName?: string
  paths: Set<string>
}

const NULL_SELECT_VALUE = '__xanesnet_null__'
const EMPTY_SIGNATURE_CONTEXT: SignatureContextValue = { paths: new Set() }
const SignatureContext = createContext<SignatureContextValue>(EMPTY_SIGNATURE_CONTEXT)

const SUMMARY_KEYS = [
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

function getInitialTheme(): ThemeMode {
  const savedTheme = window.localStorage.getItem('xanesnet-config-ui-theme')
  if (savedTheme === 'light' || savedTheme === 'dark') {
    return savedTheme
  }

  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
}

function labelFromFieldName(name: string) {
  return name.replace(/_/g, ' ')
}

function schemaTypeIncludes(schema: JsonObject | undefined, type: string) {
  if (!schema) {
    return false
  }
  return schema.type === type || (Array.isArray(schema.type) && schema.type.includes(type))
}

function schemaTypes(schema: JsonObject | undefined) {
  if (!schema) {
    return []
  }
  if (typeof schema.type === 'string') {
    return [schema.type]
  }
  if (Array.isArray(schema.type)) {
    return schema.type.filter((type): type is string => typeof type === 'string')
  }
  return []
}

function getSchemaOptions(schema: JsonObject | undefined): JsonObject[] {
  const options = Array.isArray(schema?.oneOf) ? schema.oneOf : Array.isArray(schema?.anyOf) ? schema.anyOf : undefined
  return options?.filter(isObject) ?? []
}

function fixedSchemaValue(schema: JsonObject | undefined): JsonValue | undefined {
  if (!schema) {
    return undefined
  }
  if ('const' in schema) {
    return schema.const
  }
  if (Array.isArray(schema.enum) && schema.enum.length === 1) {
    return schema.enum[0]
  }
  if ('default' in schema) {
    return schema.default
  }
  return undefined
}

function isAutoString(value: unknown) {
  return typeof value === 'string' && value.toLowerCase() === 'auto'
}

function patternMatchesAutoOnly(pattern: string) {
  try {
    const matcher = new RegExp(pattern)
    return matcher.test('auto') && matcher.test('AUTO') && !matcher.test('automatic') && !matcher.test('1')
  } catch {
    return false
  }
}

function isAutoTokenSchema(schema: JsonObject | undefined) {
  if (!schema) {
    return false
  }

  const fixedValue = fixedSchemaValue(schema)
  if (isAutoString(fixedValue)) {
    return true
  }
  if (Array.isArray(schema.enum) && schema.enum.some(isAutoString)) {
    return true
  }
  return schemaTypeIncludes(schema, 'string') && typeof schema.pattern === 'string' && patternMatchesAutoOnly(schema.pattern)
}

function isNullSchema(schema: JsonObject | undefined) {
  if (!schema) {
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

function schemaAcceptsNull(schema: JsonObject | undefined) {
  if (!schema) {
    return false
  }
  if (schemaTypeIncludes(schema, 'null')) {
    return true
  }
  if (Array.isArray(schema.enum) && schema.enum.includes(null)) {
    return true
  }
  return getSchemaOptions(schema).some(isNullSchema)
}

function nonNullSchemaForNullableSchema(schema: JsonObject): JsonObject | undefined {
  const types = schemaTypes(schema)
  if (types.includes('null')) {
    const nonNullTypes = types.filter((type) => type !== 'null')
    if (nonNullTypes.length === 0) {
      return undefined
    }
    return {
      ...schema,
      type: nonNullTypes.length === 1 ? nonNullTypes[0] : nonNullTypes,
    }
  }

  if (Array.isArray(schema.enum) && schema.enum.includes(null)) {
    return {
      ...schema,
      enum: schema.enum.filter((value) => value !== null),
    }
  }

  const options = getSchemaOptions(schema)
  if (options.length > 0 && options.some(isNullSchema)) {
    const nonNullOptions = options.filter((option) => !isNullSchema(option))
    return nonNullOptions.length === 1 ? nonNullOptions[0] : undefined
  }

  return undefined
}

function getNullableValueDefinition(schema: JsonObject | undefined): NullableValueDefinition | undefined {
  if (!schema || !schemaAcceptsNull(schema) || isNullSchema(schema)) {
    return undefined
  }

  const valueSchema = nonNullSchemaForNullableSchema(schema)
  if (!valueSchema) {
    return undefined
  }

  const enumValues = Array.isArray(valueSchema.enum) ? valueSchema.enum.filter((value) => value !== null) : undefined
  if (enumValues && enumValues.length > 0) {
    return { kind: 'enum', valueSchema, enumValues }
  }

  const types = schemaTypes(valueSchema).filter((type) => type !== 'null')
  if (types.includes('array')) {
    return {
      kind: 'array',
      valueSchema,
      itemSchema: itemSchemaForArray(valueSchema),
    }
  }
  if (types.length > 1) {
    return { kind: 'mixed', valueSchema }
  }
  if (types[0] === 'integer') {
    return { kind: 'integer', valueSchema }
  }
  if (types[0] === 'number') {
    return { kind: 'number', valueSchema }
  }
  if (types[0] === 'boolean') {
    return { kind: 'boolean', valueSchema, enumValues: [true, false] }
  }
  if (types[0] === 'string' || types.length === 0) {
    return { kind: 'string', valueSchema }
  }

  return undefined
}

function getAutoValueDefinition(schema: JsonObject | undefined): AutoValueDefinition | undefined {
  const options = getSchemaOptions(schema)
  if (options.length < 2 || !options.some(isAutoTokenSchema)) {
    return undefined
  }

  const manualSchema = options.find((option) => !isAutoTokenSchema(option))
  if (!manualSchema) {
    return undefined
  }

  if (schemaTypeIncludes(manualSchema, 'array')) {
    return { kind: 'array', manualSchema }
  }
  if (schemaTypeIncludes(manualSchema, 'integer') || schemaTypeIncludes(manualSchema, 'number')) {
    return { kind: 'number', manualSchema }
  }
  return undefined
}

function hasEntries(value: object) {
  return Object.keys(value).length > 0
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function cloneValue<T extends JsonValue | undefined>(value: T): T {
  if (value === undefined) {
    return value
  }
  return JSON.parse(JSON.stringify(value)) as T
}

function fieldPathKey(path: Array<string | number>) {
  return path.map(String).join('.')
}

function labelFromPath(path: string) {
  return path.replace(/\.(\d+)(?=\.|$)/g, '[$1]').replace(/\./g, ' > ')
}

function collectSignaturePaths(value: JsonValue, path: string[] = [], includeContainers = true): string[] {
  const paths: string[] = []
  if (path.length > 0 && includeContainers) {
    paths.push(fieldPathKey(path))
  }

  if (Array.isArray(value)) {
    value.forEach((item, index) => {
      paths.push(...collectSignaturePaths(item, [...path, String(index)], includeContainers))
    })
    if (value.length === 0 && path.length > 0 && !includeContainers) {
      paths.push(fieldPathKey(path))
    }
    return paths
  }

  if (isObject(value)) {
    const entries = Object.entries(value)
    if (entries.length === 0 && path.length > 0 && !includeContainers) {
      paths.push(fieldPathKey(path))
    }
    for (const [key, entry] of entries) {
      paths.push(...collectSignaturePaths(entry, [...path, key], includeContainers))
    }
    return paths
  }

  if (path.length > 0 && !includeContainers) {
    paths.push(fieldPathKey(path))
  }
  return paths
}

function mergeSignatureData(base: JsonValue | undefined, signature: JsonValue): JsonValue {
  if (isObject(base) && isObject(signature)) {
    if (hasDiscriminatorConflict(base, signature)) {
      return cloneValue(signature)
    }

    const merged: JsonObject = { ...base }
    for (const [key, value] of Object.entries(signature)) {
      merged[key] = mergeSignatureData(merged[key], value)
    }
    return merged
  }

  return cloneValue(signature)
}

function valuesEqual(left: JsonValue | undefined, right: JsonValue | undefined) {
  return JSON.stringify(left) === JSON.stringify(right)
}

function hasDiscriminatorConflict(base: JsonObject, signature: JsonObject) {
  return SUMMARY_KEYS.some(
    (key) => key in base && key in signature && !valuesEqual(base[key], signature[key]),
  )
}

function schemaAcceptedValues(schema: JsonObject | undefined): JsonValue[] {
  if (!schema) {
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

function valueMatchesSchema(schema: JsonObject | undefined, value: JsonValue): boolean {
  if (!schema) {
    return false
  }

  const options = getSchemaOptions(schema)
  if (options.length > 0) {
    return options.some((option) => valueMatchesSchema(option, value))
  }

  const acceptedValues = schemaAcceptedValues(schema)
  if (acceptedValues.length > 0) {
    return acceptedValues.some((acceptedValue) => valuesEqual(acceptedValue, value))
  }

  if (value === null) {
    return schemaAcceptsNull(schema)
  }
  if (Array.isArray(value)) {
    return schemaTypeIncludes(schema, 'array') || !schema.type
  }
  if (isObject(value)) {
    return schemaTypeIncludes(schema, 'object') || isObject(schema.properties) || !schema.type
  }
  if (typeof value === 'number') {
    return schemaTypeIncludes(schema, 'number') || (Number.isInteger(value) && schemaTypeIncludes(schema, 'integer')) || !schema.type
  }
  if (typeof value === 'boolean') {
    return schemaTypeIncludes(schema, 'boolean') || !schema.type
  }
  if (typeof value === 'string') {
    return schemaTypeIncludes(schema, 'string') || !schema.type
  }
  return false
}

function scoreSignatureOption(schema: JsonObject, value: JsonValue): number {
  const nestedOptions = getSchemaOptions(schema)
  if (nestedOptions.length > 0) {
    return Math.max(...nestedOptions.map((option) => scoreSignatureOption(option, value)))
  }

  if (!valueMatchesSchema(schema, value)) {
    return -1
  }
  if (!isObject(value)) {
    return 0
  }
  if (!isObject(schema.properties)) {
    return 0
  }

  let score = 0
  for (const [key, entry] of Object.entries(value)) {
    const propertySchema = schema.properties[key]
    if (!isObject(propertySchema)) {
      continue
    }

    const acceptedValues = schemaAcceptedValues(propertySchema)
    if (acceptedValues.length > 0) {
      if (acceptedValues.some((acceptedValue) => valuesEqual(acceptedValue, entry))) {
        score += SUMMARY_KEYS.includes(key) ? 100 : 8
      } else {
        return -1
      }
    } else if (valueMatchesSchema(propertySchema, entry)) {
      score += 1
    }
  }

  return score
}

function chooseSignatureOption(schema: JsonObject, value: JsonValue): JsonObject | undefined {
  const options = getSchemaOptions(schema)
  if (options.length === 0) {
    return undefined
  }

  let bestOption: JsonObject | undefined
  let bestScore = -1
  for (const option of options) {
    const score = scoreSignatureOption(option, value)
    if (score > bestScore) {
      bestOption = option
      bestScore = score
    }
  }

  return bestScore >= 0 ? bestOption : undefined
}

type SanitizedSignature = {
  value?: JsonValue
  loadedLeafPaths: string[]
  rejectedLeafPaths: string[]
}

function sanitizeSignatureForSchema(schema: JsonObject | undefined, value: JsonValue, path: string[] = []): SanitizedSignature {
  if (!schema) {
    return { rejectedLeafPaths: collectSignaturePaths(value, path, false), loadedLeafPaths: [] }
  }

  const selectedOption = chooseSignatureOption(schema, value)
  if (selectedOption) {
    return sanitizeSignatureForSchema(selectedOption, value, path)
  }
  if (getSchemaOptions(schema).length > 0) {
    return { rejectedLeafPaths: collectSignaturePaths(value, path, false), loadedLeafPaths: [] }
  }

  if (Array.isArray(value)) {
    if (!schemaTypeIncludes(schema, 'array') || !isObject(schema.items)) {
      return { rejectedLeafPaths: collectSignaturePaths(value, path, false), loadedLeafPaths: [] }
    }

    const sanitizedItems = value.map((item, index) => sanitizeSignatureForSchema(schema.items as JsonObject, item, [...path, String(index)]))
    const sanitizedArray = sanitizedItems
      .map((item) => item.value)
      .filter((item): item is JsonValue => item !== undefined)
    return {
      value: sanitizedArray,
      loadedLeafPaths: sanitizedItems.flatMap((item) => item.loadedLeafPaths),
      rejectedLeafPaths: sanitizedItems.flatMap((item) => item.rejectedLeafPaths),
    }
  }

  if (isObject(value)) {
    if (!isObject(schema.properties)) {
      return valueMatchesSchema(schema, value)
        ? { value: cloneValue(value), loadedLeafPaths: collectSignaturePaths(value, path, false), rejectedLeafPaths: [] }
        : { rejectedLeafPaths: collectSignaturePaths(value, path, false), loadedLeafPaths: [] }
    }

    const sanitizedValue: JsonObject = {}
    const loadedLeafPaths: string[] = []
    const rejectedLeafPaths: string[] = []

    for (const [key, entry] of Object.entries(value)) {
      const propertySchema = schema.properties[key]
      if (!isObject(propertySchema)) {
        rejectedLeafPaths.push(...collectSignaturePaths(entry, [...path, key], false))
        continue
      }

      const sanitizedProperty = sanitizeSignatureForSchema(propertySchema, entry, [...path, key])
      if (sanitizedProperty.value !== undefined) {
        sanitizedValue[key] = sanitizedProperty.value
      }
      loadedLeafPaths.push(...sanitizedProperty.loadedLeafPaths)
      rejectedLeafPaths.push(...sanitizedProperty.rejectedLeafPaths)
    }

    return {
      value: hasEntries(sanitizedValue) ? sanitizedValue : undefined,
      loadedLeafPaths,
      rejectedLeafPaths,
    }
  }

  return valueMatchesSchema(schema, value)
    ? { value: cloneValue(value), loadedLeafPaths: [fieldPathKey(path)], rejectedLeafPaths: [] }
    : { rejectedLeafPaths: [fieldPathKey(path)], loadedLeafPaths: [] }
}

function useSignatureContext() {
  return useContext(SignatureContext)
}

function SignatureNote() {
  return <span className="signature-note">Loaded from signature</span>
}

function mergeUiSchema(target: UiSchema, source: UiSchema) {
  const targetRecord = target as Record<string, unknown>

  for (const [key, value] of Object.entries(source)) {
    const current = targetRecord[key]
    if (!key.startsWith('ui:') && isRecord(current) && isRecord(value)) {
      mergeUiSchema(current as UiSchema, value as UiSchema)
    } else {
      targetRecord[key] = value
    }
  }

  return target
}

function buildUiSchema(schema: JsonObject | undefined): UiSchema {
  if (!schema) {
    return {}
  }

  if (getAutoValueDefinition(schema)) {
    return {
      'ui:field': 'AutoValueField',
      'ui:options': {
        fieldReplacesAnyOrOneOf: true,
      },
    }
  }

  if (getNullableValueDefinition(schema)) {
    return {
      'ui:field': 'NullableValueField',
      'ui:options': {
        fieldReplacesAnyOrOneOf: true,
      },
    }
  }

  const uiSchema: UiSchema = {}
  const uiSchemaRecord = uiSchema as Record<string, unknown>
  for (const option of getSchemaOptions(schema)) {
    mergeUiSchema(uiSchema, buildUiSchema(option))
  }

  if (isObject(schema.properties)) {
    for (const [propertyName, propertySchema] of Object.entries(schema.properties)) {
      if (!isObject(propertySchema)) {
        continue
      }
      const propertyUiSchema = buildUiSchema(propertySchema)
      if (hasEntries(propertyUiSchema)) {
        uiSchemaRecord[propertyName] = propertyUiSchema
      }
    }
  }

  if (isObject(schema.items)) {
    const itemUiSchema = buildUiSchema(schema.items)
    if (hasEntries(itemUiSchema)) {
      uiSchemaRecord.items = itemUiSchema
    }
  }

  return uiSchema
}

function createDefaultFormData(mode: ConfigMode) {
  const seed: JsonObject = mode === 'infer' ? { dataset: { num_workers: null } } : {}
  return buildCompleteConfig(SCHEMAS_BY_MODE[mode], seed)
}

function createSignatureMergeBase(currentFormData: JsonObject, signatureValue: JsonObject) {
  const base = mergeSignatureData(createDefaultFormData('infer'), currentFormData) as JsonObject
  const signatureDataset = isObject(signatureValue.dataset) ? signatureValue.dataset : undefined
  if ((!signatureDataset || !('dataset_type' in signatureDataset)) && isObject(base.dataset)) {
    delete base.dataset.dataset_type
  }
  return base
}

function sanitizeFormDataForSchema(schema: JsonObject, data: JsonObject) {
  const sanitized = sanitizeSignatureForSchema(schema, data)
  return isObject(sanitized.value) ? sanitized.value : data
}

function getPropertySchema(schema: JsonObject, propertyName: string): JsonObject | undefined {
  if (!isObject(schema.properties)) {
    return undefined
  }

  const propertySchema = schema.properties[propertyName]
  return isObject(propertySchema) ? propertySchema : undefined
}

function isFixedValueSchema(schema: JsonObject | undefined) {
  return Boolean(schema && ('const' in schema || (Array.isArray(schema.enum) && schema.enum.length === 1)))
}

function isSchemaControlledProperty(schema: JsonObject, propertyName: string) {
  return propertyName.endsWith('_type') || isFixedValueSchema(getPropertySchema(schema, propertyName))
}

function scoreSummaryOption(option: JsonObject, data: JsonValue | undefined): number {
  const nestedOptions = getSchemaOptions(option)
  if (nestedOptions.length > 0) {
    return Math.max(...nestedOptions.map((nestedOption) => scoreSummaryOption(nestedOption, data)))
  }

  if (!isObject(option.properties)) {
    return 0
  }

  const objectData = isObject(data) ? data : {}
  let score = 0

  for (const [key, propertySchema] of Object.entries(option.properties)) {
    if (!isObject(propertySchema)) {
      continue
    }

    const fixedValue = fixedSchemaValue(propertySchema)
    if (fixedValue !== undefined && key in objectData) {
      if (objectData[key] === fixedValue) {
        score += 20
      } else {
        return -1
      }
    } else if (key in objectData) {
      score += 1
    }
  }

  return score
}

function chooseSummarySchema(schema: JsonObject | undefined, data: JsonValue | undefined): JsonObject | undefined {
  const options = getSchemaOptions(schema)
  if (options.length === 0) {
    return schema
  }

  let bestOption = options[0]
  let bestScore = -1
  for (const option of options) {
    const score = scoreSummaryOption(option, data)
    if (score > bestScore) {
      bestOption = option
      bestScore = score
    }
  }
  return chooseSummarySchema(bestOption, data) ?? bestOption
}

function stringifySummaryValue(value: JsonValue | undefined) {
  if (value === null) {
    return 'null'
  }
  if (value === undefined || Array.isArray(value) || isObject(value)) {
    return undefined
  }
  return String(value)
}

function summarizeTopLevelProperty(schema: JsonObject | undefined, value: JsonValue | undefined) {
  if (Array.isArray(value)) {
    return `${value.length} item${value.length === 1 ? '' : 's'}`
  }

  const summarySchema = chooseSummarySchema(schema, value)
  const properties = isObject(summarySchema?.properties) ? summarySchema.properties : undefined
  const objectData = isObject(value) ? value : undefined

  if (properties && summarySchema) {
    for (const key of SUMMARY_KEYS) {
      const propertySchema = getPropertySchema(summarySchema, key)
      if (!propertySchema) {
        continue
      }

      const valueFromData = objectData?.[key]
      const summary = stringifySummaryValue(valueFromData ?? fixedSchemaValue(propertySchema))
      if (summary) {
        return summary
      }
    }
  }

  return stringifySummaryValue(value)
}

function isComplexTopLevelProperty(schema: JsonObject | undefined) {
  return Boolean(
    schema &&
      (schemaTypeIncludes(schema, 'object') ||
        schemaTypeIncludes(schema, 'array') ||
        Array.isArray(schema.oneOf) ||
        Array.isArray(schema.anyOf) ||
        Array.isArray(schema.allOf)),
  )
}

function itemSchemaForArray(schema: JsonObject) {
  return isObject(schema.items) ? schema.items : undefined
}

function manualDefaultValue(definition: AutoValueDefinition): JsonValue | undefined {
  const schemaDefault = fixedSchemaValue(definition.manualSchema)
  if (schemaDefault !== undefined && !isAutoString(schemaDefault)) {
    return schemaDefault
  }
  if (definition.kind === 'array') {
    return []
  }
  if (typeof definition.manualSchema.minimum === 'number') {
    return definition.manualSchema.minimum
  }
  if (typeof definition.manualSchema.exclusiveMinimum === 'number') {
    return definition.manualSchema.exclusiveMinimum + 1
  }
  return 1
}

function formatNumberArray(value: unknown) {
  return Array.isArray(value) ? value.join(', ') : ''
}

function parseNumber(value: string, integerOnly: boolean) {
  const parsed = Number(value)
  if (!Number.isFinite(parsed)) {
    return undefined
  }
  return integerOnly ? (Number.isInteger(parsed) ? parsed : undefined) : parsed
}

function parseNumberArray(value: string, integerOnly: boolean) {
  if (!value.trim()) {
    return []
  }

  return value
    .split(/[\s,]+/)
    .map((part) => parseNumber(part, integerOnly))
    .filter((part): part is number => part !== undefined)
}

function isNullInput(value: string) {
  const normalized = value.trim().toLowerCase()
  return normalized === '' || normalized === 'null'
}

function parseNullableArray(value: string, definition: NullableValueDefinition): JsonValue {
  if (isNullInput(value)) {
    return null
  }

  const itemSchema = definition.itemSchema
  const integerOnly = schemaTypeIncludes(itemSchema, 'integer')
  const numericItems = integerOnly || schemaTypeIncludes(itemSchema, 'number')
  const parts = value
    .split(numericItems ? /[\s,]+/ : ',')
    .map((part) => part.trim())
    .filter(Boolean)

  if (!numericItems) {
    return parts
  }

  return parts.map((part) => parseNumber(part, integerOnly) ?? part)
}

function parseNullableInput(value: string, definition: NullableValueDefinition): JsonValue {
  if (isNullInput(value)) {
    return null
  }

  if (definition.kind === 'array') {
    return parseNullableArray(value, definition)
  }
  if (definition.kind === 'integer' || definition.kind === 'number') {
    return parseNumber(value.trim(), definition.kind === 'integer') ?? value
  }
  if (definition.kind === 'boolean') {
    const normalized = value.trim().toLowerCase()
    if (normalized === 'true') {
      return true
    }
    if (normalized === 'false') {
      return false
    }
    return value
  }
  if (definition.kind === 'mixed') {
    const parsed = Number(value.trim())
    return Number.isFinite(parsed) ? parsed : value
  }
  return value
}

function formatNullableValue(value: unknown) {
  if (value === null || value === undefined) {
    return 'null'
  }
  if (Array.isArray(value)) {
    return value.map(String).join(', ')
  }
  if (isObject(value)) {
    return JSON.stringify(value)
  }
  return String(value)
}

function nullableSelectValue(value: unknown, definition: NullableValueDefinition) {
  if (value === null || value === undefined || !definition.enumValues) {
    return NULL_SELECT_VALUE
  }

  const index = definition.enumValues.findIndex((optionValue) => optionValue === value)
  return index >= 0 ? String(index) : NULL_SELECT_VALUE
}

function nullableInputMode(definition: NullableValueDefinition) {
  if (definition.kind === 'integer' || definition.kind === 'number') {
    return 'decimal'
  }
  if (definition.kind !== 'array') {
    return undefined
  }
  return schemaTypeIncludes(definition.itemSchema, 'integer') || schemaTypeIncludes(definition.itemSchema, 'number')
    ? 'decimal'
    : undefined
}

function SvgIcon({ name }: { name: IconName }) {
  switch (name) {
    case 'chevron-down':
      return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="m6 9 6 6 6-6" />
        </svg>
      )
    case 'chevron-up':
      return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="m18 15-6-6-6 6" />
        </svg>
      )
    case 'copy':
      return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <rect x="9" y="9" width="11" height="11" rx="2" />
          <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
        </svg>
      )
    case 'moon':
      return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M20 14.5A7.5 7.5 0 0 1 9.5 4 8.5 8.5 0 1 0 20 14.5Z" />
        </svg>
      )
    case 'plus':
      return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M12 5v14M5 12h14" />
        </svg>
      )
    case 'sun':
      return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <circle cx="12" cy="12" r="4" />
          <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41" />
        </svg>
      )
    case 'trash':
      return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M3 6h18M8 6V4h8v2M6 6l1 15h10l1-15M10 11v6M14 11v6" />
        </svg>
      )
    case 'x':
      return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M18 6 6 18M6 6l12 12" />
        </svg>
      )
  }
}

function ConfigIconButton({
  className = '',
  disabled,
  iconName,
  id,
  label,
  onClick,
  tone = 'default',
  title,
}: IconButtonProps & { iconName: IconName; label: string; tone?: 'default' | 'danger' }) {
  return (
    <button
      id={id}
      type="button"
      className={`icon-button ${tone === 'danger' ? 'danger' : ''} ${className}`.trim()}
      disabled={disabled}
      onClick={onClick}
      title={title ?? label}
      aria-label={label}
    >
      <SvgIcon name={iconName} />
      <span className="sr-only">{label}</span>
    </button>
  )
}

function AddButton(props: IconButtonProps) {
  return (
    <div className="array-add-control">
      <ConfigIconButton {...props} iconName="plus" label="Add" />
    </div>
  )
}

function CopyButton(props: IconButtonProps) {
  return <ConfigIconButton {...props} iconName="copy" label="Copy" />
}

function MoveDownButton(props: IconButtonProps) {
  return <ConfigIconButton {...props} iconName="chevron-down" label="Move down" />
}

function MoveUpButton(props: IconButtonProps) {
  return <ConfigIconButton {...props} iconName="chevron-up" label="Move up" />
}

function RemoveButton(props: IconButtonProps) {
  return <ConfigIconButton {...props} iconName="trash" label="Remove" tone="danger" />
}

function ClearButton(props: IconButtonProps) {
  return <ConfigIconButton {...props} iconName="x" label="Clear" />
}

function AutoValueField(props: FieldProps) {
  const schema = props.schema as JsonObject
  const definition = getAutoValueDefinition(schema)
  const [manualModeOverride, setManualModeOverride] = useState<boolean | undefined>(undefined)
  const [numberText, setNumberText] = useState<string | undefined>(undefined)
  const [arrayText, setArrayText] = useState<string | undefined>(undefined)

  if (!definition) {
    return null
  }

  const autoDefinition = definition
  const manualMode = manualModeOverride ?? (props.formData !== undefined && !isAutoString(props.formData))
  const disabled = props.disabled || props.readonly
  const manualSchema = autoDefinition.manualSchema
  const itemSchema = itemSchemaForArray(manualSchema)
  const integerOnly = autoDefinition.kind === 'array' ? schemaTypeIncludes(itemSchema, 'integer') : schemaTypeIncludes(manualSchema, 'integer')
  const minimum = autoDefinition.kind === 'array' ? itemSchema?.minimum : manualSchema.minimum
  const labelId = `${props.fieldPathId.$id}-label`

  function emit(value: JsonValue | undefined) {
    props.onChange(value, props.fieldPathId.path, undefined, props.fieldPathId.$id)
  }

  function selectAuto() {
    setManualModeOverride(false)
    emit('auto')
  }

  function selectManual() {
    setManualModeOverride(true)
    if (props.formData === undefined || isAutoString(props.formData)) {
      const fallback = manualDefaultValue(autoDefinition)
      if (typeof fallback === 'number') {
        setNumberText(String(fallback))
      } else if (Array.isArray(fallback)) {
        setArrayText(formatNumberArray(fallback))
      }
      emit(fallback)
    }
  }

  return (
    <div className="auto-value-field">
      <div id={labelId} className="auto-value-label">
        {labelFromFieldName(props.name)}
        {props.required ? '*' : ''}
      </div>
      <div className="auto-value-modes" role="group" aria-labelledby={labelId}>
        <button
          type="button"
          className={manualMode ? 'auto-value-mode' : 'auto-value-mode active'}
          aria-pressed={!manualMode}
          disabled={disabled}
          onClick={selectAuto}
        >
          Auto
        </button>
        <button
          type="button"
          className={manualMode ? 'auto-value-mode active' : 'auto-value-mode'}
          aria-pressed={manualMode}
          disabled={disabled}
          onClick={selectManual}
        >
          {autoDefinition.kind === 'array' ? 'List' : 'Number'}
        </button>
      </div>
      {manualMode && autoDefinition.kind === 'number' ? (
        <input
          id={props.fieldPathId.$id}
          name={props.name}
          type="number"
          min={typeof minimum === 'number' ? minimum : undefined}
          step={integerOnly ? 1 : 'any'}
          value={numberText ?? (typeof props.formData === 'number' ? String(props.formData) : '')}
          disabled={disabled}
          onBlur={() => props.onBlur(props.fieldPathId.$id, props.formData)}
          onFocus={() => props.onFocus(props.fieldPathId.$id, props.formData)}
          onChange={(event) => {
            const nextValue = event.currentTarget.value
            setNumberText(nextValue)
            emit(nextValue === '' ? undefined : parseNumber(nextValue, integerOnly))
          }}
        />
      ) : null}
      {manualMode && autoDefinition.kind === 'array' ? (
        <input
          id={props.fieldPathId.$id}
          name={props.name}
          type="text"
          inputMode="numeric"
          value={arrayText ?? formatNumberArray(props.formData)}
          disabled={disabled}
          placeholder="1, 2, 3"
          onBlur={() => props.onBlur(props.fieldPathId.$id, props.formData)}
          onFocus={() => props.onFocus(props.fieldPathId.$id, props.formData)}
          onChange={(event) => {
            const nextValue = event.currentTarget.value
            setArrayText(nextValue)
            emit(parseNumberArray(nextValue, integerOnly))
          }}
        />
      ) : null}
    </div>
  )
}

function NullableValueField(props: FieldProps) {
  const schema = props.schema as JsonObject
  const definition = getNullableValueDefinition(schema)
  const [inputText, setInputText] = useState(() => formatNullableValue(props.formData))

  if (!definition) {
    return null
  }

  const disabled = props.disabled || props.readonly
  const labelId = `${props.fieldPathId.$id}-label`
  const description = typeof schema.description === 'string' ? schema.description : undefined

  function emit(value: JsonValue) {
    props.onChange(value, props.fieldPathId.path, undefined, props.fieldPathId.$id)
  }

  return (
    <div className="nullable-value-field">
      <label id={labelId} className="nullable-value-label" htmlFor={props.fieldPathId.$id}>
        {labelFromFieldName(props.name)}
        {props.required ? '*' : ''}
      </label>
      {definition.enumValues ? (
        <select
          id={props.fieldPathId.$id}
          name={props.name}
          value={nullableSelectValue(props.formData, definition)}
          disabled={disabled}
          onBlur={() => props.onBlur(props.fieldPathId.$id, props.formData)}
          onFocus={() => props.onFocus(props.fieldPathId.$id, props.formData)}
          onChange={(event) => {
            const selectedValue = event.currentTarget.value
            if (selectedValue === NULL_SELECT_VALUE) {
              emit(null)
              return
            }
            const selectedIndex = Number(selectedValue)
            emit(definition.enumValues?.[selectedIndex] ?? null)
          }}
        >
          <option value={NULL_SELECT_VALUE}>null</option>
          {definition.enumValues.map((value, index) => (
            <option key={`${props.fieldPathId.$id}-${index}`} value={String(index)}>
              {String(value)}
            </option>
          ))}
        </select>
      ) : (
        <input
          id={props.fieldPathId.$id}
          name={props.name}
          type="text"
          inputMode={nullableInputMode(definition)}
          value={inputText}
          disabled={disabled}
          placeholder="null"
          onBlur={() => {
            setInputText(formatNullableValue(parseNullableInput(inputText, definition)))
            props.onBlur(props.fieldPathId.$id, props.formData)
          }}
          onFocus={() => props.onFocus(props.fieldPathId.$id, props.formData)}
          onChange={(event) => {
            const nextText = event.currentTarget.value
            setInputText(nextText)
            emit(parseNullableInput(nextText, definition))
          }}
        />
      )}
      {description ? <p className="field-description">{description}</p> : null}
    </div>
  )
}

function ObjectFieldTemplate(props: ObjectFieldTemplateProps) {
  const { className, description, disabled, fieldPathId, formData, onAddProperty, properties, readonly, required, schema, title } =
    props
  const objectSchema = schema as JsonObject
  const signatureContext = useSignatureContext()
  const isRootObject = fieldPathId.path.length === 0
  const isPureUnionSchema = (schema.oneOf || schema.anyOf) && !schema.properties && properties.length === 0

  if (isPureUnionSchema) {
    return null
  }

  const addButton = canExpand(schema, props.uiSchema, formData) ? (
    <AddButton
      id={buttonId(fieldPathId, 'add')}
      className="rjsf-object-property-expand"
      onClick={onAddProperty}
      disabled={disabled || readonly}
      registry={props.registry}
    />
  ) : null

  function renderProperty(property: ObjectFieldProperty) {
    const propertyPath = fieldPathKey([...fieldPathId.path, property.name])
    const isFromSignature = signatureContext.paths.has(propertyPath)

    if (property.hidden) {
      return property.content
    }

    const propertySchema = getPropertySchema(objectSchema, property.name)

    if (isSchemaControlledProperty(objectSchema, property.name)) {
      if (!isFixedValueSchema(propertySchema) && fixedSchemaValue(propertySchema) === undefined) {
        return null
      }

      return (
        <div key={property.name} className="schema-controlled-field" hidden>
          {property.content}
        </div>
      )
    }

    return isFromSignature ? (
      <div key={property.name} className="signature-field" data-signature-path={propertyPath}>
        {property.content}
        <SignatureNote />
      </div>
    ) : (
      property.content
    )
  }

  const visibleProperties = properties.filter(
    (property) => !property.hidden && !isSchemaControlledProperty(objectSchema, property.name),
  )
  const renderedProperties = properties.map(renderProperty)

  if (isRootObject) {
    return (
      <div className="rjsf-root-object" id={fieldPathId.$id}>
        {properties.map((property) => {
          if (property.hidden) {
            return property.content
          }

          const propertySchema = getPropertySchema(objectSchema, property.name)

          if (!isComplexTopLevelProperty(propertySchema)) {
            return <div key={property.name}>{renderProperty(property)}</div>
          }

          const sectionSummary = summarizeTopLevelProperty(
            propertySchema,
            isObject(formData) ? formData[property.name] : undefined,
          )
          const propertyPath = fieldPathKey([...fieldPathId.path, property.name])
          const isFromSignature = signatureContext.paths.has(propertyPath)

          return (
            <details
              key={`${fieldPathId.$id}-${property.name}`}
              className={isFromSignature ? 'top-level-section signature-section' : 'top-level-section'}
            >
              <summary>
                <span className="top-level-section-heading">
                  <span className="top-level-section-title">{labelFromFieldName(property.name)}</span>
                  {sectionSummary ? <span className="top-level-section-summary">{sectionSummary}</span> : null}
                  {isFromSignature ? <span className="signature-section-note">Signature</span> : null}
                </span>
                <SvgIcon name="chevron-down" />
              </summary>
              <div className="top-level-section-body">{renderProperty(property)}</div>
            </details>
          )
        })}
        {addButton}
      </div>
    )
  }

  if (visibleProperties.length === 0 && !addButton) {
    return <>{renderedProperties}</>
  }

  return (
    <fieldset className={className} id={fieldPathId.$id}>
      {title ? (
        <legend>
          {title}
          {required ? '*' : ''}
        </legend>
      ) : null}
      {description ? <p className="field-description">{description}</p> : null}
      {renderedProperties}
      {addButton}
    </fieldset>
  )
}

const rjsfTemplates = {
  ObjectFieldTemplate,
  ButtonTemplates: {
    AddButton,
    ClearButton,
    CopyButton,
    MoveDownButton,
    MoveUpButton,
    RemoveButton,
  },
}

const rjsfFields = {
  AutoValueField,
  NullableValueField,
}

const rjsfDefaultFormStateBehavior = {
  emptyObjectFields: 'populateRequiredDefaults',
  constAsDefaults: 'always',
} as const

function App() {
  const [mode, setMode] = useState<ConfigMode>('train')
  const [theme, setTheme] = useState<ThemeMode>(getInitialTheme)
  const [formInstanceKey, setFormInstanceKey] = useState(0)
  const [formData, setFormData] = useState<JsonObject>(() => createDefaultFormData('train'))
  const [signatureState, setSignatureState] = useState<SignatureState | null>(null)
  const [status, setStatus] = useState<Status>({
    tone: 'idle',
    message: 'Choose a config type and edit the form to preview the YAML config.',
  })

  useEffect(() => {
    document.documentElement.dataset.theme = theme
    window.localStorage.setItem('xanesnet-config-ui-theme', theme)
  }, [theme])

  const activeOption = useMemo(
    () => CONFIG_SCHEMA_OPTIONS.find((option) => option.id === mode) ?? CONFIG_SCHEMA_OPTIONS[0],
    [mode],
  )
  const schema = activeOption.schema
  const uiSchema = useMemo(() => buildUiSchema(schema), [schema])
  const yamlOutput = useMemo(() => {
    const completeConfig = buildCompleteConfig(schema, formData)
    return formatYamlConfig(completeConfig, schema, mode)
  }, [formData, mode, schema])
  const signatureContext = useMemo<SignatureContextValue>(() => {
    if (!signatureState) {
      return EMPTY_SIGNATURE_CONTEXT
    }
    return {
      fileName: signatureState.fileName,
      paths: new Set(signatureState.paths),
    }
  }, [signatureState])

  function switchMode(nextMode: ConfigMode) {
    setMode(nextMode)
    setFormInstanceKey((currentKey) => currentKey + 1)
    setFormData(createDefaultFormData(nextMode))
    setSignatureState(null)
    setStatus({ tone: 'idle', message: `Started a new ${nextMode} config.` })
  }

  async function loadExistingConfig(file: File | undefined) {
    if (!file) {
      return
    }

    try {
      const parsed = loadYaml(await file.text())
      if (!isObject(parsed)) {
        throw new Error('The YAML file must contain a top-level mapping.')
      }

      const detectedMode = detectConfigMode(parsed)
      const detectedSchema = SCHEMAS_BY_MODE[detectedMode]
      setMode(detectedMode)
      setFormInstanceKey((currentKey) => currentKey + 1)
      setFormData(buildCompleteConfig(detectedSchema, parsed))
      setSignatureState(null)
      setStatus({ tone: 'success', message: `Loaded ${file.name} as a ${detectedMode} config.` })
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Could not parse the YAML file.'
      setStatus({ tone: 'error', message })
    }
  }

  async function loadSignature(file: File | undefined) {
    if (!file) {
      return
    }

    try {
      const parsed = loadYaml(await file.text())
      if (!isObject(parsed)) {
        throw new Error('The signature YAML file must contain a top-level mapping.')
      }

      const sanitizedSignature = sanitizeSignatureForSchema(SCHEMAS_BY_MODE.infer, parsed)
      const signatureValue = isObject(sanitizedSignature.value) ? sanitizedSignature.value : {}
      const allSignaturePaths = collectSignaturePaths(signatureValue)
      const merged = mergeSignatureData(createSignatureMergeBase(formData, signatureValue), signatureValue)
      const sanitizedMerged = sanitizeSignatureForSchema(SCHEMAS_BY_MODE.infer, isObject(merged) ? merged : signatureValue)
      const mergedValue = isObject(sanitizedMerged.value) ? sanitizedMerged.value : signatureValue
      const nextFormData = buildCompleteConfig(SCHEMAS_BY_MODE.infer, mergedValue)

      setMode('infer')
      setFormInstanceKey((currentKey) => currentKey + 1)
      setFormData(nextFormData)
      setSignatureState({
        fileName: file.name,
        paths: allSignaturePaths,
        leafPaths: sanitizedSignature.loadedLeafPaths,
        schemaExtraLeafPaths: sanitizedSignature.rejectedLeafPaths,
      })
      setStatus({
        tone: sanitizedSignature.rejectedLeafPaths.length > 0 ? 'error' : 'success',
        message:
          sanitizedSignature.rejectedLeafPaths.length > 0
            ? `Loaded ${file.name}, but ${sanitizedSignature.rejectedLeafPaths.length} signature field(s) are outside the current schemas.`
            : `Loaded ${file.name} as an inference signature.`,
      })
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Could not parse the signature YAML file.'
      setStatus({ tone: 'error', message })
    }
  }

  function markValidConfig() {
    setStatus({ tone: 'success', message: 'Config validates against the current schema.' })
  }

  function downloadYaml() {
    if (!yamlOutput) {
      return
    }

    const blob = new Blob([yamlOutput], { type: 'text/yaml;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `xanesnet-${mode}-config.yaml`
    link.click()
    URL.revokeObjectURL(url)
  }

  return (
    <main className="app-shell">
      <header className="app-header">
        <div>
          <p className="eyebrow">XANESNET config UI</p>
          <h1>XANESNET config creator</h1>
        </div>
        <div className="header-actions">
          <button
            type="button"
            className="theme-toggle"
            onClick={() => setTheme((currentTheme) => (currentTheme === 'dark' ? 'light' : 'dark'))}
            aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
          >
            <SvgIcon name={theme === 'dark' ? 'sun' : 'moon'} />
            <span>{theme === 'dark' ? 'Light' : 'Dark'}</span>
          </button>
          <label className="file-loader">
            <span>Load YAML</span>
            <input
              type="file"
              accept=".yaml,.yml,text/yaml,application/x-yaml"
              onChange={(event) => {
                void loadExistingConfig(event.currentTarget.files?.[0])
                event.currentTarget.value = ''
              }}
            />
          </label>
          {mode === 'infer' ? (
            <label className="file-loader signature-loader">
              <span>Load signature</span>
              <input
                type="file"
                accept=".yaml,.yml,text/yaml,application/x-yaml"
                onChange={(event) => {
                  void loadSignature(event.currentTarget.files?.[0])
                  event.currentTarget.value = ''
                }}
              />
            </label>
          ) : null}
        </div>
      </header>

      <section className="mode-bar" aria-label="Config type">
        {CONFIG_SCHEMA_OPTIONS.map((option) => (
          <button
            key={option.id}
            type="button"
            className={option.id === mode ? 'mode-button active' : 'mode-button'}
            onClick={() => switchMode(option.id)}
          >
            <span>{option.label}</span>
            <small>{option.description}</small>
          </button>
        ))}
      </section>

      <section className={`status ${status.tone}`} role="status">
        {status.message}
      </section>

      <div className="workspace-grid">
        <section className="editor-panel" aria-label="Configuration form">
          <div className="panel-heading">
            <h2>{activeOption.label} config</h2>
            <p>Fields, defaults, and compatibility rules come from the bundled JSON Schemas.</p>
          </div>
          {signatureState ? (
            <section className="signature-summary" aria-label="Loaded signature">
              <div>
                <strong>{signatureState.fileName}</strong>
                <span>{signatureState.leafPaths.length} field{signatureState.leafPaths.length === 1 ? '' : 's'} loaded</span>
              </div>
              {signatureState.schemaExtraLeafPaths.length > 0 ? (
                <details>
                  <summary>{signatureState.schemaExtraLeafPaths.length} outside current schemas</summary>
                  <ul>
                    {signatureState.schemaExtraLeafPaths.map((path) => (
                      <li key={path}>{labelFromPath(path)}</li>
                    ))}
                  </ul>
                </details>
              ) : null}
            </section>
          ) : null}
          <SignatureContext.Provider value={signatureContext}>
            <Form
              key={`${mode}-${formInstanceKey}`}
              schema={schema}
              uiSchema={uiSchema}
              validator={validator}
              formData={formData}
              fields={rjsfFields}
              templates={rjsfTemplates}
              experimental_defaultFormStateBehavior={rjsfDefaultFormStateBehavior}
              omitExtraData
              liveOmit="onChange"
              noHtml5Validate
              showErrorList="top"
              onChange={(event: IChangeEvent) => {
                const nextFormData = (event.formData ?? {}) as JsonObject
                setFormData(sanitizeFormDataForSchema(schema, nextFormData))
              }}
              onSubmit={() => markValidConfig()}
              onError={() => setStatus({ tone: 'error', message: 'Fix the validation errors before using this YAML.' })}
            >
              <div className="form-actions">
                <button type="submit" className="primary-action">
                  Validate config
                </button>
                <button
                  type="button"
                  className="secondary-action"
                  onClick={() => {
                    setFormInstanceKey((currentKey) => currentKey + 1)
                    setFormData(createDefaultFormData(mode))
                    setSignatureState(null)
                    setStatus({ tone: 'idle', message: `Cleared the ${mode} form.` })
                  }}
                >
                  Clear
                </button>
              </div>
            </Form>
          </SignatureContext.Provider>
        </section>

        <aside className="output-panel" aria-label="Live YAML preview">
          <div className="panel-heading">
            <h2>Live YAML</h2>
          </div>
          <textarea
            className="yaml-output"
            value={yamlOutput}
            readOnly
            spellCheck={false}
            aria-label="Live YAML config preview"
          />
          <div className="output-actions">
            <button type="button" className="secondary-action" disabled={!yamlOutput} onClick={downloadYaml}>
              Download YAML
            </button>
            <button
              type="button"
              className="secondary-action"
              disabled={!yamlOutput}
              onClick={() => void navigator.clipboard.writeText(yamlOutput)}
            >
              Copy
            </button>
          </div>
        </aside>
      </div>

      <footer className="app-footer">XANESNET, Hendrik Junkawitsch config creator</footer>
    </main>
  )
}

export default App
