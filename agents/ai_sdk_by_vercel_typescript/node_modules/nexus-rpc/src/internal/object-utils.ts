/**
 * Creates a new object by mapping all string properties (keys and values) of an existing object
 * through a mapper function.
 *
 * @param obj - The source object, whose properties will be mapped.
 * @param fn - The mapper function.
 *
 * @returns A new object with the properties mapped to the new type.
 *
 * @internal
 * @hidden
 */
export function mapKeyValues<T extends Record<string, any>, U>(
  obj: T,
  fn: (key: keyof T & string, value: T[keyof T & string]) => U,
): Record<keyof T, U> {
  return Object.fromEntries(
    Object.entries(obj).map(([key, value]) => [key, fn(key, value)]),
  ) as Record<keyof T, U>;
}
