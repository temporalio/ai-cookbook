import { Content } from "./content";

/**
 * Serializer is used by the framework to serialize/deserialize input and output.
 *
 * @experimental
 */
export interface Serializer {
  /**
   * Serialize encodes a value into a {@link Content}.
   */
  serialize(value: unknown): Content;

  /**
   * Deserialize decodes a {@link Content} into a value.
   */
  deserialize<T = unknown>(content: Content): T;
}
