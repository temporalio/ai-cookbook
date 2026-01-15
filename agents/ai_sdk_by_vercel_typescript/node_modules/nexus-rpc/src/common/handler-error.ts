import { injectSymbolBasedInstanceOf } from "../internal/symbol-instanceof";

/**
 * A Nexus handler error.
 *
 * This error class represents an error that occurred during the handling of a
 * Nexus operation that should be reported to the caller as a handler error.
 *
 * Example:
 *
 * ```ts
 *     import { HandlerError } from "nexus-rpc";
 *
 *     // Throw a bad request error
 *     throw new HandlerError("BAD_REQUEST", "Invalid input provided");
 *
 *     // Throw a bad request error, with a cause
 *     throw new HandlerError("BAD_REQUEST", "Invalid input provided", { cause });
 *
 *     // Throw a retryable internal error
 *     throw new HandlerError("INTERNAL", "Database unavailable", { retryableOverride: true });
 * ```
 *
 * @experimental
 */
export class HandlerError extends Error {
  /**
   * One of the predefined error types.
   *
   * @see {@link HandlerErrorType}
   */
  public readonly type: HandlerErrorType;

  /**
   * Whether this error should be considered retryable.
   *
   * By default, the retry behavior is determined from the error type.
   * For example, by default, `INTERNAL` is retryable, but `UNAVAILABLE` is non-retryable.
   *
   * If specified, `retryableOverride` overrides the default retry behavior determined based on
   * the error type. Use {@link retryable} to determine the effective retry behavior.
   *
   * @see {@link retryable}.
   */
  public readonly retryableOverride: boolean | undefined;

  /**
   * Constructs a new {@link HandlerError}.
   *
   * @param type - The type of the error.
   * @param message - The message of the error.
   * @param options - Extra options for the error, including the cause and retryable override.
   *
   * @experimental
   */
  constructor(type: HandlerErrorType, message?: string | undefined, options?: HandlerErrorOptions) {
    const actualMessage = message || `Handler error: ${type}`;

    super(actualMessage, { cause: options?.cause });
    this.type = type;
    this.retryableOverride = options?.retryableOverride;
  }

  /**
   * Whether this error is retryable.
   *
   * This differs from the {@link retryableOverride} property in that `retryable` takes into
   * account the default behavior resulting from the error type, if no override is provided.
   *
   * @see {@link retryableOverride}.
   */
  public get retryable(): boolean {
    if (typeof this.retryableOverride === "boolean") return this.retryableOverride;

    switch (this.type) {
      case "BAD_REQUEST":
      case "UNAUTHENTICATED":
      case "UNAUTHORIZED":
      case "NOT_FOUND":
      case "NOT_IMPLEMENTED":
        return false;

      case "UNAVAILABLE":
      case "UPSTREAM_TIMEOUT":
      case "RESOURCE_EXHAUSTED":
      case "INTERNAL":
        return true;

      default: {
        // Force a compile time error if missing a case
        const _noMissingCase: never = this.type;
        return true;
      }
    }
  }
}

injectSymbolBasedInstanceOf(HandlerError, "HandlerError");

/**
 * Options for constructing a {@link HandlerError}.
 *
 * @experimental
 * @inline
 */
export interface HandlerErrorOptions {
  /**
   * Underlying cause of the error.
   */
  cause?: unknown;

  /**
   * Whether this error should be considered retryable.
   *
   * If not set, the retry behavior is determined from the error type.
   * For example, by default, `INTERNAL` is retryable, but `UNAVAILABLE` is non-retryable.
   */
  retryableOverride?: boolean | undefined;
}

/**
 * An error type associated with a {@link HandlerError}, defined according to the Nexus specification.
 *
 * @experimental
 */
export type HandlerErrorType = (typeof HandlerErrorType)[keyof typeof HandlerErrorType];
export const HandlerErrorType = {
  /**
   * The handler cannot or will not process the request due to an apparent client error.
   *
   * Clients should not retry this request unless advised otherwise.
   */
  BAD_REQUEST: "BAD_REQUEST",

  /**
   * The client did not supply valid authentication credentials for this request.
   *
   * Clients should not retry this request unless advised otherwise.
   */
  UNAUTHENTICATED: "UNAUTHENTICATED",

  /**
   * The caller does not have permission to execute the specified operation.
   *
   * Clients should not retry this request unless advised otherwise.
   */
  UNAUTHORIZED: "UNAUTHORIZED",

  /**
   * The requested resource could not be found but may be available in the future.
   */
  NOT_FOUND: "NOT_FOUND",

  /**
   * Some resource has been exhausted, perhaps a per-user quota, or perhaps the entire file system
   * is out of space.
   *
   * Subsequent requests by the client are permissible.
   */
  RESOURCE_EXHAUSTED: "RESOURCE_EXHAUSTED",

  /**
   * An internal error occured.
   *
   * Subsequent requests by the client are permissible.
   */
  INTERNAL: "INTERNAL",

  /**
   * The server either does not recognize the request method, or it lacks the ability to fulfill the
   * request. Clients should not retry this request unless advised otherwise.
   */
  NOT_IMPLEMENTED: "NOT_IMPLEMENTED",

  /**
   * The service is currently unavailable.
   *
   * Subsequent requests by the client are permissible.
   */
  UNAVAILABLE: "UNAVAILABLE",

  /**
   * Used by gateways to report that a request to an upstream server has timed out.
   *
   * Subsequent requests by the client are permissible.
   */
  UPSTREAM_TIMEOUT: "UPSTREAM_TIMEOUT",
} as const;
