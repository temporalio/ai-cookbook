/** @import { OperationHandler } from "./operation-handler" */

/**
 * An internal symbol, used to prevent direct implementation of interfaces.
 *
 * @hidden
 * @internal
 */
const isHandlerStartOperationResultSymbol = Symbol("__nexus_isHandlerStartOperationResult");

/**
 * The return type from the {@link OperationHandler.start} method.
 *
 * Use either {@link HandlerStartOperationResult.sync} or {@link HandlerStartOperationResult.async}
 * to create a result object. Do not implement this interface directly.
 *
 * @experimental
 */
export type HandlerStartOperationResult<T = unknown> =
  | HandlerStartOperationResultSync<T>
  | HandlerStartOperationResultAsync;

/**
 * The return type from the {@link OperationHandler.start} method.
 *
 * Use either {@link HandlerStartOperationResult.sync} or {@link HandlerStartOperationResult.async}
 * to create a result object. Do not implement this interface directly.
 *
 * @experimental
 */
export const HandlerStartOperationResult = {
  /**
   * Create a result that indicates that an operation has been accepted and will complete asynchronously.
   */
  async(token: string): HandlerStartOperationResultAsync {
    return {
      isAsync: true,
      token,
      [isHandlerStartOperationResultSymbol]: true,
    };
  },

  /**
   * Create a result that indicates that an operation completed successfully.
   */
  sync<T>(value: T): HandlerStartOperationResultSync<T> {
    return {
      isAsync: false,
      value,
      [isHandlerStartOperationResultSymbol]: true,
    };
  },

  [Symbol.hasInstance]: function (this: any, value: object): boolean {
    return (
      typeof value === "object" &&
      value !== null &&
      (value as any)[isHandlerStartOperationResultSymbol] === true
    );
  },
};

/**
 * A result that indicates that an operation completed successfully.
 *
 * Use {@link HandlerStartOperationResult.sync} to create a sync result object.
 * Do not implement this interface directly.
 *
 * @example
 * ```typescript
 *   return HandlerStartOperationResult.sync(42);
 * ```
 *
 * @experimental
 */
export interface HandlerStartOperationResultSync<T = unknown> {
  /**
   * Indicate whether the operation completed synchronously (false) or will complete
   * asynchronously (true).
   */
  isAsync: false;

  /**
   * The return value of the operation.
   */
  value: T;

  /**
   * Prevents direct implementation of this interface.
   *
   * @hidden
   * @internal
   */
  [isHandlerStartOperationResultSymbol]: true;
}

/**
 * A result that indicates that an operation has been accepted and will complete asynchronously.
 *
 * Use {@link HandlerStartOperationResult.async} to create an async result object.
 * Do not implement this interface directly.
 *
 * @example
 * ```typescript
 *   return HandlerStartOperationResult.async("unique token");
 * ```
 *
 * @experimental
 */
export interface HandlerStartOperationResultAsync {
  /**
   * Indicate whether the operation completed synchronously (false) or will complete
   * asynchronously (true).
   */
  isAsync: true;

  /**
   * A token to identify the operation in followup handler methods such as {@link OperationHandler.getResult}
   * and {@link OperationHandler.cancel}.
   */
  token: string;

  /**
   * Prevents direct implementation of this interface.
   *
   * @hidden
   * @internal
   */
  [isHandlerStartOperationResultSymbol]: true;
}
