export declare const inputBrand: unique symbol;
export declare const outputBrand: unique symbol;

/**
 * Definition of a Nexus service contract, including its name and operations.
 *
 * Can only be constructed by the {@link service} function.
 *
 * @experimental
 */
export interface ServiceDefinition<Ops extends OperationMap = OperationMap> {
  name: string;
  operations: Ops;
}

/**
 * An operation contract that describes the name, and input and output types of an operation.
 *
 * @experimental
 */
export interface OperationDefinition<I, O> {
  name: string;
  [inputBrand]: I;
  [outputBrand]: O;
}

/**
 * A named collection of operations, as defined by a {@link ServiceDefinition}.
 *
 * @experimental
 */
export type OperationMap = Record<string, OperationDefinition<any, any>>;

/**
 * A mapped type that extracts the input type from an operation in a service.
 *
 * @experimental
 */
export type OperationInput<T> = T extends OperationDefinition<infer I, any> ? I : any;

/**
 * A mapped type that extracts the output type from an operation in a service.
 *
 * @experimental
 */
export type OperationOutput<T> = T extends OperationDefinition<any, infer O> ? O : any;

/**
 * A mapped type that extracts all operation names from a service.
 *
 * @experimental
 */
export type OperationKey<T> = {
  [K in keyof T & string]: T[K] extends OperationDefinition<any, any> ? K : never;
}[keyof T & string];

/**
 * Confirm that a service definition is valid.
 *
 * @param service - The service definition to validate.
 *
 * @throws {TypeError} If the service definition is invalid.
 *
 * @experimental
 */
export function validateServiceDefinition(service: ServiceDefinition) {
  if (typeof service.name !== "string" || !service.name) {
    throw new TypeError("Service name must be a non-empty string");
  }

  const operationNames = new Set<string>();
  for (const [propName, operation] of Object.entries(service.operations)) {
    const operationName = operation.name;
    if (typeof operationName !== "string" || !operationName) {
      throw new TypeError(`Operation name must be a non-empty string, for property '${propName}'`);
    }
    if (operationNames.has(operationName)) {
      throw new TypeError(`Duplicate operation definition for name: '${operationName}'`);
    }
    operationNames.add(operationName);
  }
}
