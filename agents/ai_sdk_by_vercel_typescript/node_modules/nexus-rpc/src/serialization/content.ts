/**
 * A container for a map of headers and a byte array of data.
 *
 * It is used by the SDK's {@link Serializer} interface implementations.
 *
 * @experimental
 */
export interface Content {
  /**
   * Header that should include information on how to deserialize this content.
   * Headers constructed by the framework always have lower case keys.
   * User provided keys are considered case-insensitive by the framework.
   */
  headers: Record<string, string>;

  /**
   * Request or response data. May be undefined for empty data.
   */
  data?: Uint8Array;
}
