"""Inventory optimization and replenishment recommendation module.

Exports core supply chain calculations:
- calculate_safety_stock: Buffer sizing under forecast error and lead time uncertainty.
- calculate_lead_time_demand: Sum of daily demand forecasts across lead time.
- calculate_reorder_point: Replenishment trigger threshold (LTD + Safety Stock).
- calculate_target_stock: Periodic review order-up-to stocking level (P = L + R).
- calculate_order_quantity: Discrete non-negative replenishment sizing.
- calculate_economic_order_quantity: Classical Wilson EOQ cost trade-off benchmark.
- inventory_position: Net stock position (On-Hand + On-Order - Backorders).
- estimate_forecast_uncertainty: Out-of-sample error variance estimation with fallback.
- generate_inventory_recommendations: End-to-end multi-product inventory decision engine.
- generate_replenishment_orders: Filtered actionable replenishment purchase orders.

Operational Boundary Disclaimer:
All stock levels (on-hand, on-order, backorders), lead times, and operational costs
are SIMULATED parameters. The underlying UCI dataset does not record real warehouse
inventory or supplier contracts.
"""

from retail_forecasting.inventory.calculate_stock import (
    calculate_safety_stock,
    calculate_lead_time_demand,
    calculate_protection_period_demand,
    calculate_reorder_point,
    calculate_target_stock,
    calculate_order_quantity,
    calculate_economic_order_quantity,
)

from retail_forecasting.inventory.reorder_recommendations import (
    inventory_position,
    evaluate_inventory_position,
    estimate_forecast_uncertainty,
    generate_inventory_recommendations,
    generate_replenishment_orders,
)

__all__ = [
    "calculate_safety_stock",
    "calculate_lead_time_demand",
    "calculate_protection_period_demand",
    "calculate_reorder_point",
    "calculate_target_stock",
    "calculate_order_quantity",
    "calculate_economic_order_quantity",
    "inventory_position",
    "evaluate_inventory_position",
    "estimate_forecast_uncertainty",
    "generate_inventory_recommendations",
    "generate_replenishment_orders",
]
