/**
 * Flatten the type output to improve type hints shown in editors.
 *
 * This utility type forces TypeScript into performing a "flattening" operation at a specific point
 * in the type evaluation tree, which may sometime result in more readable type hints in editors.
 *
 * See https://github.com/sindresorhus/type-fest/blob/main/source/simplify.d.ts
 *
 * @internal
 */
export type Simplify<T> = { [K in keyof T]: T[K] } & {};

/**
 * Defines a type that is a union of literal string values, but that also accepts any other string,
 * without sacrificing auto-completion in TypeScript editors for the literal type part of the union.
 *
 * See https://github.com/sindresorhus/type-fest/blob/main/source/literal-union.d.ts
 *
 * @internal
 */
export type LiteralStringUnion<T> = T | (string & {});
