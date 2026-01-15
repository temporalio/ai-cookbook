import { HandlerError, OperationInfo } from "../common";
import { LazyValue } from "../serialization";
import { HandlerStartOperationResult } from "./start-operation-result";
import {
  OperationContext,
  StartOperationContext,
  GetOperationResultContext,
  GetOperationInfoContext,
  CancelOperationContext,
} from "./operation-context";
import { ServiceHandler } from "./service-handler";
import { CompiledOperationHandlerFor } from "./operation-handler";

/**
 * The root Nexus handler, which dispatches Nexus requests to a collection of registered service
 * implementations.
 *
 * @experimental
 */
export class ServiceRegistry {
  /**
   * Constructs a new {@link ServiceRegistry}.
   *
   * @experimental
   */
  public static create(services: ServiceHandler<any>[]) {
    const serviceMap = new Map<string, ServiceHandler>();

    for (const s of services) {
      const name = s.definition.name;
      if (!name) {
        throw new TypeError("Tried to register a Nexus service with no name");
      }
      if (serviceMap.has(name)) {
        throw new TypeError(`Duplicate registration of nexus service '${name}'`);
      }
      serviceMap.set(name, s);
    }

    return new ServiceRegistry(serviceMap);
  }

  private constructor(
    /**
     * Registered service handlers to which this registry dispatches requests.
     */
    private readonly services = new Map<string, ServiceHandler>(),
  ) {}

  private getOperationHandler(ctx: OperationContext): CompiledOperationHandlerFor<any> {
    const { service, operation } = ctx;
    const serviceHandler = this.services.get(service);
    if (serviceHandler == null) {
      throw new HandlerError(
        "NOT_FOUND",
        `No service handler registered for service name '${service}'`,
      );
    }
    return serviceHandler.getOperationHandler(operation);
  }

  async start(
    ctx: StartOperationContext,
    lv: LazyValue,
  ): Promise<HandlerStartOperationResult<any>> {
    const handler = this.getOperationHandler(ctx);
    const input = await lv.consume<any>();
    return await handler.start(ctx, input);
  }

  async getInfo(ctx: GetOperationInfoContext, token: string): Promise<OperationInfo> {
    return await this.getOperationHandler(ctx).getInfo(ctx, token);
  }

  async getResult(ctx: GetOperationResultContext, token: string): Promise<any> {
    return await this.getOperationHandler(ctx).getResult(ctx, token);
  }

  async cancel(ctx: CancelOperationContext, token: string): Promise<void> {
    return await this.getOperationHandler(ctx).cancel(ctx, token);
  }
}
