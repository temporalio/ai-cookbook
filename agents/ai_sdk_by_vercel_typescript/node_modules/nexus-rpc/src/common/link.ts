/**
 * Link contains a URL and a type that can be used to decode the URL.
 * It can be used to pass information about the caller to the handler, or vice-versa.
 *
 * @experimental
 */
export interface Link {
  /**
   * An arbitrary URL.
   */
  readonly url: URL;

  /**
   * Type information for decoding the URL.
   *
   * Valid chars: alphanumeric, '_', '.', '/'.
   */
  readonly type: string;
}
