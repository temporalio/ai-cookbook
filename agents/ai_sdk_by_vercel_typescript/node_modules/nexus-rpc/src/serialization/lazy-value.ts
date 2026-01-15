import { injectSymbolBasedInstanceOf } from "../internal/symbol-instanceof";
import { Serializer } from "./serializer";

/**
 * A container for a value encoded in an underlying stream.
 * It is used to stream inputs and outputs in the various client and server APIs.
 *
 * @experimental
 */
export class LazyValue {
  /**
   * @experimental
   */
  constructor(
    readonly serializer: Serializer,

    /**
     * Headers that should include information on how to process the stream's content.
     * Headers constructed by the framework always have lower case keys.
     * User provided keys are considered case-insensitive by the framework.
     */
    readonly headers: Record<string, string>,

    /**
     * ReadableStream that contains request or response data. May be undefined for empty data.
     */
    public readonly stream?: ReadableStream<Uint8Array>,
  ) {}

  /**
   * Consume the underlying reader stream, deserializing via the embedded serializer.
   */
  async consume<T = unknown>(): Promise<T> {
    if (this.stream == null) {
      // Return a default value from the serializer.
      return this.serializer.deserialize({ headers: this.headers });
    }
    const reader = this.stream.getReader();
    const chunks = Array<Uint8Array>();
    let length = 0;

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      chunks.push(value);
      length += value.length;
    }

    const data = new Uint8Array(length);
    let offset = 0;
    for (const chunk of chunks) {
      data.set(chunk, offset);
      offset += chunk.length;
    }

    return this.serializer.deserialize<T>({ headers: this.headers, data });
  }
}

injectSymbolBasedInstanceOf(LazyValue, "LazyValue");
