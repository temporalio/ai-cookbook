import { HandlerError } from "../common";
import { OperationDefinition, OperationKey, OperationMap, ServiceDefinition } from "../service";
import {
  CompiledOperationHandlerFor,
  compileOperationHandler,
  OperationHandler,
  SyncOperationHandler,
} from "./operation-handler";

/**
 * A Nexus Service implementation, that provides handlers for each of its operations.
 *
 * @experimental
 */
export class ServiceHandler<Ops extends OperationMap = OperationMap> {
  /**
   * Build a `ServiceHandler` from a service definition and a collection of operation handlers.
   *
   * There must be an operation handler for every operation in the service definition.
   *
   * @param service The service definition
   * @param handlers The collection of handlers
   * @returns The compiled service handler
   *
   * @internal
   * @hidden
   */
  static build<Ops extends OperationMap>(
    service: ServiceDefinition<Ops>,
    handlers: ServiceHandlerFor<Ops>,
  ): ServiceHandler<Ops> {
    const operations = new Map<OperationKey<Ops>, CompiledOperationHandlerFor<Ops>>();

    for (const [propName, definition] of Object.entries(service.operations)) {
      const compiledOpHandler = compileOperationHandler(definition, handlers[propName]);
      const operationName = compiledOpHandler.name as OperationKey<Ops>;

      if (operations.has(operationName)) {
        throw new TypeError(
          `Operation with name '${operationName}' already registered for service '${service.name}'`,
        );
      }

      operations.set(operationName, compiledOpHandler);
    }

    return new ServiceHandler(service, handlers, operations);
  }

  private constructor(
    public readonly definition: ServiceDefinition<Ops>,
    public readonly handlers: ServiceHandlerFor<Ops>,
    private readonly operationsMap: Map<OperationKey<Ops>, CompiledOperationHandlerFor<Ops>>,
  ) {}

  /**
   * Returns the definition and handler for a given operation.
   *
   * @param operationName
   */
  public getOperationHandler<K extends OperationKey<Ops>>(
    operationName: K,
  ): CompiledOperationHandlerFor<Ops> {
    const entry = this.operationsMap.get(operationName);
    if (entry == null) {
      throw new HandlerError(
        "NOT_FOUND",
        `Operation handler not registered for operation '${operationName}' in service '${this.definition.name}'`,
      );
    }
    return entry;
  }
}

/**
 * Constructs a service handler for a given service contract.
 *
 * @experimental
 */
export function serviceHandler<Ops extends OperationMap>(
  service: ServiceDefinition<Ops>,
  handlers: ServiceHandlerFor<Ops>,
): ServiceHandler<Ops> {
  return ServiceHandler.build(service, handlers);
}

/**
 * A type that defines a handler for a given operation.
 *
 * @experimental
 */
export type OperationHandlerFor<T> =
  T extends OperationDefinition<infer I, infer O>
    ? OperationHandler<I, O> | SyncOperationHandler<I, O>
    : never;

/**
 * A type that defines a collection of handlers for a given collection of operation interfaces.
 *
 * @experimental
 */
export type ServiceHandlerFor<T extends OperationMap = OperationMap> = {
  [K in keyof T & string]: OperationHandlerFor<T[K]>;
};
