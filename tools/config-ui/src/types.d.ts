declare module 'js-yaml' {
  export interface DumpOptions {
    condenseFlow?: boolean
    flowLevel?: number
    forceQuotes?: boolean
    indent?: number
    lineWidth?: number
    noArrayIndent?: boolean
    noCompatMode?: boolean
    noRefs?: boolean
    quotingType?: '"' | "'"
    sortKeys?: boolean | ((a: string, b: string) => number)
  }

  export function load(input: string): unknown
  export function dump(input: unknown, options?: DumpOptions): string
}