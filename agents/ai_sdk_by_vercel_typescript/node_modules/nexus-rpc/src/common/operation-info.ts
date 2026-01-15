/**
 * Information about an operation.
 *
 * @experimental
 */
export interface OperationInfo {
  /**
   * Token for the operation.
   */
  readonly token: string;

  /**
   * State of the operation.
   */
  readonly state: OperationState;
}

/**
 * Describes the current state of an operation.
 *
 * @experimental
 */
export type OperationState = "succeeded" | "failed" | "canceled" | "running";
