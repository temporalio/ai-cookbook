import { HandlerError, OperationInfo } from "../common";
import { HandlerStartOperationResult } from "./start-operation-result";
import {
  CancelOperationContext,
  GetOperationInfoContext,
  GetOperationResultContext,
  StartOperationContext,
} from "./operation-context";
import {
  OperationDefinition,
  OperationInput,
  OperationKey,
  OperationMap,
  OperationOutput,
} from "../service";

/**
 * A handler for a Nexus operation.
 *
 * This interface is meant to be implemented by Nexus service implementors.
 *
 * @experimental
 */
export interface OperationHandler<I, O> {
  /**
   * Handle requests to start an operation.
   *
   * Return {@link HandlerStartOperationResultSync} to respond successfully inline, or
   * {@link HandlerStartOperationResultAsync} to indicate that an asynchronous operation was started.
   * Throw an {@link OperationError} to indicate that an operation completed as failed or canceled.
   */
  start(ctx: StartOperationContext, input: I): Promise<HandlerStartOperationResult<O>>;

  /**
   * Handle requests to get information about an asynchronous operation.
   */
  getInfo(ctx: GetOperationInfoContext, token: string): Promise<OperationInfo>;

  /**
   * Handle requests to get the result of an asynchronous operation. Return non error result to
   * respond successfully inline, or throw an {@link OperationStillRunningError} to indicate that an
   * asynchronous operation is still running.
   *
   * Throw an {@link OperationError} to indicate that an operation completed as failed or canceled.
   *
   * When {@link GetOperationResultContext.timeoutMs | timeoutMs} is greater than zero, this request
   * should be treated as a long poll. Note that the specified wait duration may be longer than the
   * configured client or server side request timeout, and should be handled separately.
   *
   * It is the implementor's responsibility to respect the client's timeout duration and return in a
   * timely fashion, leaving enough time for the request to complete and the response to be sent
   * back.
   */
  getResult(ctx: GetOperationResultContext, token: string): Promise<O>;

  /**
   * Handle requests to cancel an asynchronous operation.
   *
   * Cancelation of a Nexus operation is:
   * 1. _asynchronous_ - returning from this method only confirms that cancelation was notified;
   *    the implementation may however choose to process the cancellation at a later time, or to
   *    ignore it entirely.
   * 2. _idempotent_ - implementations must ignore duplicate cancelations for the same operation.
   */
  cancel(ctx: CancelOperationContext, token: string): Promise<void>;
}

/**
 * A shortcut for defining an operation handler that only implements the {@link OperationHandler.start}
 * method and always returns a {@link HandlerStartOperationResultSync}.
 *
 * @experimental
 */
export type SyncOperationHandler<I, O> = (ctx: StartOperationContext, input: I) => Promise<O>;

/**
 * Compiles an operation handler into a {@link CompiledOperationHandler}. A compiled operation
 * handler is a single object that is both an operation definition and a full-fledged operation
 * handler for that operation.
 *
 * @hidden
 * @internal
 */
export function compileOperationHandler<I, O>(
  definition: OperationDefinition<I, O>,
  handler: OperationHandler<I, O> | SyncOperationHandler<I, O> | undefined,
): CompiledOperationHandler<I, O> {
  if (handler == null) {
    throw new TypeError(
      `No handler registered for operation '${definition.name}' (expected property name '${definition.name}')`,
    );
  }

  if (typeof handler === "function") {
    // Operation handler is declared using the shortcut syntax. Wrap it into a full-fledged handler.
    return {
      ...definition,

      start: async (ctx, input) => {
        return HandlerStartOperationResult.sync(await handler(ctx, input));
      },
      getInfo: notImplemented,
      getResult: notImplemented,
      cancel: notImplemented,
    };
  }

  if (typeof handler.start !== "function") {
    throw new TypeError(`Handler for operation '${definition.name}' has no start method`);
  }

  return {
    ...definition,

    // Defensively ensure that the handler has all the required methods,
    // defaulting to throwing a not implemented error if some methods are missing.
    start: handler.start.bind(handler),
    getInfo: handler.getInfo?.bind(handler) ?? notImplemented,
    getResult: handler.getResult?.bind(handler) ?? notImplemented,
    cancel: handler.cancel?.bind(handler) ?? notImplemented,
  };
}

/**
 * A compiled operation handler is a single object that is both an operation definition and a
 * full-fledged operation handler for that operation.
 *
 * @hidden
 * @internal
 * @experimental
 */
export type CompiledOperationHandler<I, O> = OperationDefinition<I, O> & OperationHandler<I, O>;

/**
 * @internal
 * @hidden
 */
export type CompiledOperationHandlerFor<Ops extends OperationMap> = CompiledOperationHandler<
  OperationInput<Ops[OperationKey<Ops>]>,
  OperationOutput<Ops[OperationKey<Ops>]>
>;

/**
 * @internal
 * @hidden
 */
function notImplemented(): never {
  throw new HandlerError("NOT_IMPLEMENTED", "Not implemented");
}
